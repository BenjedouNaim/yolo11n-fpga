"""Post-training int8 quantization of YOLO11n, matched to the FPGA partitioning.

Scheme (what the conv IP will implement):
  * weights:      int8, symmetric, per output channel     w = s_w[c] * w_q
  * activations:  int8, asymmetric, per tensor            x = s * (x_q - zp)
  * bias:         int32, scale s_in * s_w[c]
  * accumulator:  int32; activation (SiLU) applied after rescaling, output re-quantized to int8
  * Detect's last 1x1 convs (box/class logits) keep their int32 output (dequantized on the CPU),
    and the DFL / NMS post-processing stays in float on the CPU.

The model is evaluated with fake quantization (quantize -> dequantize in float), which reproduces
the integer arithmetic up to float rounding of the int32 accumulator.

Outputs:
  data/quant/<conv>.wq.bin      int8 weights   [C_out, C_in/groups, K, K]
  data/quant/<conv>.wscale.bin  float32 per-channel weight scale [C_out]
  data/quant/qparams.txt        qconv <name> <in_scale> <in_zp> <out_scale> <out_zp>  (out_scale 0 = int32 output)
  reports/quantization.md       float vs int8 mAP for each calibration method

Usage:
  python python/quantize.py [--calib 64] [--data coco128.yaml]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_golden import IMG_EXTS, letterbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
QMIN, QMAX = -128, 127


class RangeObserver:
    """Collects activation ranges: global min/max and per-batch percentiles."""

    def __init__(self, pct):
        self.pct = pct
        self.lo, self.hi = float("inf"), float("-inf")
        self.plo, self.phi = [], []

    def __call__(self, x):
        x = x.detach().float().flatten()
        self.lo, self.hi = min(self.lo, x.min().item()), max(self.hi, x.max().item())
        sample = x[torch.randint(0, x.numel(), (min(x.numel(), 200_000),))]
        q = torch.quantile(sample, torch.tensor([1 - self.pct / 100, self.pct / 100]))
        self.plo.append(q[0].item())
        self.phi.append(q[1].item())

    def range(self, method):
        if method == "minmax":
            return self.lo, self.hi
        return float(np.mean(self.plo)), float(np.mean(self.phi))


def act_qparams(lo, hi):
    """Asymmetric int8 scale / zero point covering [lo, hi] (always including 0)."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = max((hi - lo) / (QMAX - QMIN), 1e-8)
    zp = int(np.clip(round(QMIN - lo / scale), QMIN, QMAX))
    return scale, zp


class FakeQuant(nn.Module):
    def __init__(self, scale, zp):
        super().__init__()
        self.scale, self.zp = float(scale), int(zp)

    def forward(self, x):
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zp, QMIN, QMAX)


class QConv2d(nn.Module):
    """nn.Conv2d with int8 input, int8 per-channel weights and int32 bias (fake-quantized)."""

    def __init__(self, conv, in_scale, in_zp):
        super().__init__()
        self.conv = conv
        self.in_q = FakeQuant(in_scale, in_zp)
        w = conv.weight.detach()
        self.w_scale = (w.abs().amax(dim=(1, 2, 3)) / 127).clamp(min=1e-8)
        self.w_q = torch.round(w / self.w_scale.view(-1, 1, 1, 1)).clamp(-127, 127)
        if conv.bias is not None:
            b_scale = in_scale * self.w_scale
            self.b_fq = torch.round(conv.bias.detach() / b_scale) * b_scale
        else:
            self.b_fq = None

    def forward(self, x):
        w = self.w_q * self.w_scale.view(-1, 1, 1, 1)
        c = self.conv
        return nn.functional.conv2d(self.in_q(x), w, self.b_fq, c.stride, c.padding, c.dilation, c.groups)


def load_fused(weights):
    model = YOLO(str(weights))
    model.model.fuse(verbose=False)
    model.model.eval()
    return model


def quant_targets(net):
    """(conv_name, nn.Conv2d, parent Conv module or None) for every conv that runs on the IP."""
    parents = {f"{n}.conv": m for n, m in net.named_modules() if hasattr(m, "conv") and hasattr(m, "act")}
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and not name.endswith(".dfl.conv"):  # DFL is CPU post-processing
            yield name, m, parents.get(name)


def calibrate(net, images, imgsz, pct):
    obs_in, obs_out, hooks = {}, {}, []
    for name, conv, parent in quant_targets(net):
        obs_in[name] = RangeObserver(pct)
        hooks.append(conv.register_forward_pre_hook(lambda m, inp, o=obs_in[name]: o(inp[0])))
        if parent is not None:
            obs_out[name] = RangeObserver(pct)
            hooks.append(parent.register_forward_hook(lambda m, inp, out, o=obs_out[name]: o(out)))
    with torch.no_grad():
        for p in images:
            boxed, _, _ = letterbox(cv2.imread(str(p)), imgsz)
            x = torch.from_numpy(boxed[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
            net(x)
    for h in hooks:
        h.remove()
    return obs_in, obs_out


def apply_quant(net, obs_in, obs_out, method):
    """Swap convs for QConv2d and add output fake-quant after each Conv's activation."""
    qparams = {}
    for name, conv, parent in list(quant_targets(net)):
        s_in, zp_in = act_qparams(*obs_in[name].range(method))
        q = QConv2d(conv, s_in, zp_in)
        owner = net.get_submodule(name.rsplit(".", 1)[0])
        setattr(owner, name.rsplit(".", 1)[1], q)
        s_out, zp_out = 0.0, 0
        if parent is not None:
            s_out, zp_out = act_qparams(*obs_out[name].range(method))
            parent.act = nn.Sequential(parent.act, FakeQuant(s_out, zp_out))
        qparams[name] = (q, s_in, zp_in, s_out, zp_out)
    return qparams


def export_quant(qparams, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, (q, s_in, zp_in, s_out, zp_out) in qparams.items():
        key = name.replace(".", "_")
        q.w_q.to(torch.int8).numpy().tofile(out_dir / f"{key}.wq.bin")
        q.w_scale.numpy().astype("<f4").tofile(out_dir / f"{key}.wscale.bin")
        lines.append(f"qconv {name} {s_in!r} {zp_in} {s_out!r} {zp_out}")
    (out_dir / "qparams.txt").write_text("\n".join(lines) + "\n")


def evaluate(model, data, imgsz):
    m = model.val(data=data, imgsz=imgsz, batch=8, device="cpu", plots=False, verbose=False, workers=0)
    return m.box.map50, m.box.map


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="coco128.yaml", help="Ultralytics dataset yaml used for calibration and mAP")
    ap.add_argument("--calib", type=int, default=64, help="number of calibration images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--methods", default="minmax,pct99.99,pct99.999")
    args = ap.parse_args()

    weights = DATA / "weights" / "yolo11n.pt"
    ds = check_det_dataset(args.data)
    train_dir = Path(ds["train"]) if not isinstance(ds["train"], list) else Path(ds["train"][0])
    images = sorted(p for p in train_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)[: args.calib]
    print(f"calibration: {len(images)} images from {train_dir}")

    results = [("float32", *evaluate(load_fused(weights), args.data, args.imgsz))]
    print(f"float32: mAP50 {results[0][1]:.4f}  mAP50-95 {results[0][2]:.4f}")

    best = None
    for method in args.methods.split(","):
        model = load_fused(weights)
        pct = float(method[3:]) if method.startswith("pct") else 99.99
        obs_in, obs_out = calibrate(model.model, images, args.imgsz, pct)
        qparams = apply_quant(model.model, obs_in, obs_out, "minmax" if method == "minmax" else "pct")
        map50, map_ = evaluate(model, args.data, args.imgsz)
        print(f"int8 {method}: mAP50 {map50:.4f}  mAP50-95 {map_:.4f}")
        results.append((f"int8 ({method})", map50, map_))
        if best is None or map_ > best[1]:
            best = (method, map_, qparams)

    export_quant(best[2], DATA / "quant")
    print(f"exported int8 params ({best[0]}) -> {DATA / 'quant'}")

    f_map50, f_map = results[0][1], results[0][2]
    rows = "\n".join(f"| {n} | {a * 100:.1f} | {b * 100:.1f} | {(b - f_map) * 100:+.1f} |" for n, a, b in results)
    report = (
        "# Quantization results\n\n"
        f"Dataset: `{args.data}` (val split), imgsz {args.imgsz}, calibration on {len(images)} train images.\n"
        f"Exported calibration: **{best[0]}**.\n\n"
        "| Model | mAP50 (%) | mAP50-95 (%) | Δ mAP50-95 vs float (pts) |\n| --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "Scheme: int8 per-channel symmetric weights, int8 per-tensor asymmetric activations, int32 bias and "
        "accumulator; Detect logits kept as int32; DFL and NMS in float on the CPU.\n\n"
        "Caveat: coco128 has only 128 images and its train and val splits are the same images, so calibration "
        "and evaluation overlap and differences under ~1 point are noise. Confirm on COCO val2017 (5000 images) "
        "for the final report.\n"
    )
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "quantization.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
