"""Export YOLO11n weights and per-layer golden outputs for the C++ / HLS flow.

All .bin files are raw float32, little-endian, C-contiguous (NCHW for activations,
OIHW for conv weights). BatchNorm is fused into the conv weights/bias, so the C++
model only needs conv + bias + activation.

Outputs:
  data/weights/yolo11n.pt                  downloaded once by Ultralytics
  data/weights/<conv>.weight.bin / .bias.bin
  data/weights/manifest.json               conv shapes/params + layer graph (type, from)
  data/golden/<image>/input.bin            1x3xSxS, RGB, 0..1, letterboxed
  data/golden/<image>/layer_XX_<Type>.bin  output of each top-level layer
  data/golden/<image>/manifest.json        shapes, letterbox params, detections
  data/golden/<image>/preview.jpg          image with the golden boxes drawn

Usage:
  python python/export_golden.py [--imgsz 640] [--conf 0.25] [--iou 0.45]
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import ASSETS

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # older Ultralytics releases
    from ultralytics.utils.ops import non_max_suppression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def save(t, path):
    """Write a tensor as raw float32 and return its shape."""
    a = np.ascontiguousarray(t.detach().cpu().float().numpy()).astype("<f4")
    a.tofile(path)
    return list(a.shape)


def letterbox(img, size, color=114):
    """Resize keeping aspect ratio, pad to size x size. Returns image, ratio, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def export_weights(net, out_dir):
    # Ultralytics Conv/DWConv wrap an nn.Conv2d (".conv") plus an activation (".act")
    act_of = {f"{n}.conv": type(m.act).__name__ for n, m in net.named_modules() if hasattr(m, "conv") and hasattr(m, "act")}

    convs, lines = [], []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            k, s, p = m.kernel_size, m.stride, m.padding
            assert k[0] == k[1] and s[0] == s[1] and p[0] == p[1] and m.dilation == (1, 1), name
            act = act_of.get(name, "Identity")
            assert act in ("SiLU", "Identity"), f"{name}: unsupported activation {act}"
            key = name.replace(".", "_")
            save(m.weight, out_dir / f"{key}.weight.bin")
            if m.bias is not None:
                save(m.bias, out_dir / f"{key}.bias.bin")
            c_in = m.weight.shape[1] * m.groups
            convs.append({
                "name": name,
                "file": key,
                "shape": list(m.weight.shape),  # [C_out, C_in/groups, K, K]
                "stride": s[0],
                "padding": p[0],
                "groups": m.groups,
                "act": act,
                "bias": m.bias is not None,
            })
            # conv <name> <c_out> <c_in> <k> <stride> <pad> <groups> <silu> <bias>
            lines.append(f"conv {name} {m.out_channels} {c_in} {k[0]} {s[0]} {p[0]} {m.groups} "
                         f"{int(act == 'SiLU')} {int(m.bias is not None)}")
        if isinstance(getattr(m, "add", None), bool):  # Bottleneck / PSABlock / SPPF shortcut
            lines.append(f"flag {name} add {int(m.add)}")
        if type(m).__name__ == "Attention":
            lines.append(f"attn {name} {m.num_heads} {m.key_dim} {m.head_dim} {m.scale!r}")

    # Top-level graph: "from" = -1 means previous layer, a list means concat/detect inputs
    layers = [{"i": i, "type": type(m).__name__, "from": m.f} for i, m in enumerate(net.model)]
    detect = net.model[-1]
    manifest = {"convs": convs, "layers": layers, "names": net.names,
                "strides": detect.stride.tolist(), "reg_max": detect.reg_max, "nc": detect.nc}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines.append(f"detect {detect.nc} {detect.reg_max} " + " ".join(str(int(s)) for s in detect.stride.tolist()))
    for layer in layers:  # layer <i> <Type> <from...>
        f = layer["from"] if isinstance(layer["from"], list) else [layer["from"]]
        lines.append(f"layer {layer['i']} {layer['type']} " + " ".join(str(v) for v in f))
    (out_dir / "weights.txt").write_text("\n".join(lines) + "\n")
    return len(convs), len(layers)


def export_golden(net, img_path, out_dir, imgsz, conf, iou):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxed, r, (pad_x, pad_y) = letterbox(img, imgsz)
    x = torch.from_numpy(boxed[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
    input_shape = save(x, out_dir / "input.bin")

    outputs = []

    def save_output(prefix, out):
        # Detect returns (decoded, [P3, P4, P5]); nested outputs get _k suffixes
        if isinstance(out, torch.Tensor):
            outputs.append({"file": f"{prefix}.bin", "shape": save(out, out_dir / f"{prefix}.bin")})
        elif isinstance(out, (list, tuple)):
            for k, o in enumerate(out):
                save_output(f"{prefix}_{k}", o)

    hooks = [
        m.register_forward_hook(lambda mod, inp, out, i=i: save_output(f"layer_{i:02d}_{type(mod).__name__}", out))
        for i, m in enumerate(net.model)
    ]
    with torch.no_grad():
        pred = net(x)
    for hk in hooks:
        hk.remove()

    # Final detections: NMS in letterbox space, then map back to the original image
    p = pred[0] if isinstance(pred, (list, tuple)) else pred
    det = non_max_suppression(p, conf_thres=conf, iou_thres=iou)[0]
    boxes = det[:, :4].clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / r).clamp(0, w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / r).clamp(0, h)

    detections = []
    for b, c, k in zip(boxes.tolist(), det[:, 4].tolist(), det[:, 5].int().tolist()):
        detections.append({"box_xyxy": [round(v, 1) for v in b], "conf": round(c, 4), "cls": k, "name": net.names[k]})
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{net.names[k]} {c:.2f}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / "preview.jpg"), img)

    manifest = {
        "image": img_path.name,
        "orig_hw": [h, w],
        "imgsz": imgsz,
        "ratio": r,
        "pad_xy": [pad_x, pad_y],
        "input": {"file": "input.bin", "shape": input_shape},
        "layers": outputs,
        "conf": conf,
        "iou": iou,
        "detections": detections,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Plain-text copies for the C++ side: letterbox params and golden detections
    (out_dir / "meta.txt").write_text(f"{h} {w} {r!r} {pad_x} {pad_y} {conf} {iou}\n")
    (out_dir / "detections.txt").write_text("".join(
        f"{d['cls']} {d['conf']} {' '.join(str(v) for v in d['box_xyxy'])}\n" for d in detections))
    return len(outputs), len(detections)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    img_dir, w_dir, g_dir = DATA / "images", DATA / "weights", DATA / "golden"
    for d in (img_dir, w_dir, g_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        # Ultralytics ships two sample images; add your own to data/images for a third
        for name in ("bus.jpg", "zidane.jpg"):
            shutil.copy(ASSETS / name, img_dir / name)
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    model = YOLO(str(w_dir / "yolo11n.pt"))
    net = model.model
    net.fuse(verbose=False)
    net.eval()

    n_conv, n_layers = export_weights(net, w_dir)
    print(f"weights: {n_conv} convs, {n_layers} layers -> {w_dir}")

    for img_path in images:
        n_out, n_det = export_golden(net, img_path, g_dir / img_path.stem, args.imgsz, args.conf, args.iou)
        print(f"golden:  {img_path.name}: {n_out} tensors, {n_det} detections -> {g_dir / img_path.stem}")


if __name__ == "__main__":
    main()
