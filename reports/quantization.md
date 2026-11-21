# Quantization results

Dataset: `coco128.yaml` (val split), imgsz 640, calibration on 64 train images.
Exported calibration: **pct99.999**.

| Model | mAP50 (%) | mAP50-95 (%) | Δ mAP50-95 vs float (pts) |
| --- | --- | --- | --- |
| float32 | 67.1 | 50.2 | +0.0 |
| int8 (minmax) | 67.6 | 50.6 | +0.4 |
| int8 (pct99.99) | 67.0 | 49.7 | -0.5 |
| int8 (pct99.999) | 67.7 | 50.9 | +0.6 |

Scheme: int8 per-channel symmetric weights, int8 per-tensor asymmetric activations, int32 bias and accumulator; Detect logits kept as int32; DFL and NMS in float on the CPU.

Caveat: coco128 has only 128 images and its train and val splits are the same images, so calibration and evaluation overlap and differences under ~1 point are noise. Confirm on COCO val2017 (5000 images) for the final report.
