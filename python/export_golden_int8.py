"""Golden outputs of the int8 (fake-quantized) model, for the HLS testbench.

Re-applies the exported quantization (data/quant/qparams.txt) to the fused model, runs it on
each data/golden/<image>/input.bin and writes, next to the float golden files:
  layer_XX_<Type>_int8.bin     output of each top-level layer
  layer_23_Detect_0_int8.bin   decoded Detect output [1, 4 + nc, A]
  detections_int8.txt          cls conf x1 y1 x2 y2 (original image pixels)

Usage:
  python python/export_golden_int8.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_golden import non_max_suppression, save  # noqa: E402
from quantize import FakeQuant, QConv2d, load_fused, quant_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def apply_qparams(net, path):
    qp = {}
    for line in path.read_text().splitlines():
        _, name, s_in, zp_in, s_out, zp_out = line.split()
        qp[name] = (float(s_in), int(zp_in), float(s_out), int(zp_out))
    for name, conv, parent in list(quant_targets(net)):
        s_in, zp_in, s_out, zp_out = qp[name]
        owner = net.get_submodule(name.rsplit(".", 1)[0])
        setattr(owner, name.rsplit(".", 1)[1], QConv2d(conv, s_in, zp_in))
        if parent is not None and s_out > 0:
            parent.act = nn.Sequential(parent.act, FakeQuant(s_out, zp_out))


def main():
    model = load_fused(DATA / "weights" / "yolo11n.pt")
    net = model.model
    apply_qparams(net, DATA / "quant" / "qparams.txt")

    def saver(path):
        def hook(mod, inp, out):
            save(out, path)  # returns the shape; the hook must return None or it replaces the output
        return hook

    for gdir in sorted(p for p in (DATA / "golden").iterdir() if p.is_dir()):
        h, w, r, pad_x, pad_y, conf, iou = (gdir / "meta.txt").read_text().split()
        h, w, r, pad_x, pad_y, conf, iou = int(h), int(w), float(r), float(pad_x), float(pad_y), float(conf), float(iou)
        x = np.fromfile(gdir / "input.bin", dtype="<f4")
        s = int(round((x.size / 3) ** 0.5))
        x = torch.from_numpy(x.reshape(1, 3, s, s))

        hooks = [m.register_forward_hook(saver(gdir / f"layer_{i:02d}_{type(m).__name__}_int8.bin"))
                 for i, m in enumerate(net.model[:-1])]
        with torch.no_grad():
            pred = net(x)
        for hk in hooks:
            hk.remove()
        p = pred[0] if isinstance(pred, (list, tuple)) else pred
        save(p, gdir / "layer_23_Detect_0_int8.bin")

        det = non_max_suppression(p, conf_thres=conf, iou_thres=iou)[0]
        boxes = det[:, :4].clone()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / r).clamp(0, w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / r).clamp(0, h)
        lines = [f"{k} {round(c, 4)} " + " ".join(str(round(v, 1)) for v in b)
                 for b, c, k in zip(boxes.tolist(), det[:, 4].tolist(), det[:, 5].int().tolist())]
        (gdir / "detections_int8.txt").write_text("".join(l + "\n" for l in lines))
        print(f"{gdir.name}: {len(lines)} int8 detections")


if __name__ == "__main__":
    main()
