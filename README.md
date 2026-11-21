# YOLO11n on FPGA (KV260) — HW/SW co-design

Phase 1: simulation-level implementation in Vitis HLS (no board).
Phase 2: Vivado integration, Linux on KV260, runtime testing.

## Layout

| Folder | Contents |
| --- | --- |
| `python/` | `export_golden.py`, `quantize.py` |
| `data/weights/` | Exported layer weights (`.bin`) |
| `data/golden/` | Per-layer PyTorch outputs (golden reference) |
| `data/images/` | Test images |
| `cpp_ref/` | Plain C++ reference model of YOLO11n (float) |
| `hls/src/` | HLS IP source (`conv_ip.cpp`, `conv_ip.h`) |
| `hls/tb/` | Testbench acting as the processor (`tb.cpp`) |
| `reports/` | csim/cosim logs, synthesis reports, waveforms |
| `ip_export/` | Exported IP (`.zip`) for Vivado |
| `docs/` | Report, slides, demo images |

## How to run

1. Python environment (on D:, CPU PyTorch is enough):
   ```
   python -m venv .venv
   .venv\Scripts\pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
   .venv\Scripts\pip install --no-cache-dir -r requirements.txt
   ```
2. Golden reference (downloads `yolo11n.pt` once, exports fused weights + per-layer outputs):
   ```
   .venv\Scripts\python python\export_golden.py
   ```
3. C++ float reference model, compared layer by layer with PyTorch:
   ```
   cpp_ref\build.bat
   cd cpp_ref && build\yolo_ref.exe ..\data
   ```

4. int8 quantization (downloads coco128 once, ~7 MB; exports to `data/quant/`):
   ```
   .venv\Scripts\python python\quantize.py
   ```

5. int8 golden from PyTorch (per-layer outputs + detections of the fake-quantized model):
   ```
   .venv\Scripts\python python\export_golden_int8.py
   ```
6. HLS IP, C simulation without Vitis (MSVC): every conv runs through `conv_ip`
   ```
   hls\tb\build_csim.bat v1
   hls\tb\build\tb_v1.exe data
   ```
   (`v0` builds the baseline the same way.)
7. HLS IP with Vitis HLS 2025.2 (from `hls/`, cmd): csim (unit test) + synthesis, then co-simulation + IP export
   ```
   set HLS_VER=v1& set HLS_COSIM=1& D:\Xilinx\2025.2\Vitis\bin\vitis-run.bat --mode hls --tcl script.tcl
   ```
   (`HLS_VER=v0` for the baseline, `HLS_COSIM=0` to stop after synthesis, `HLS_TRACE=1` for waveforms.)

## Status

| Step | State |
| --- | --- |
| 1. Golden reference | Done: 88 convs, 24 layers, 2 sample images (add a 3rd to `data/images`) |
| 2. C++ float reference | Done: all 24 layers and final detections match PyTorch (max rel. err < 1e-4) |
| 3. Quantization | Done: int8 PTQ, mAP50-95 50.9% vs 50.2% float on coco128 (see `reports/quantization.md`) |
| 4. HLS conv IP + testbench | csim PASS (MSVC): IP bit-exact vs software int8 conv on 174 calls; detections match PyTorch int8 (IoU >= 0.97) |
| 5. Synthesis + optimization | Done: both meet 200 MHz (3.65 ns); v1 uses 68 BRAM / 79 DSP / 18k LUT and is 11.3x faster than v0 in co-simulation (`reports/optimization.md`) |
| 6. Co-simulation + IP export | Done: C/RTL co-sim PASS for v0 and v1 (bit-exact); IPs in `ip_export/conv_ip_v0.zip`, `conv_ip_v1.zip` |
| Phase 2 | Next: Vivado block design with Zynq UltraScale+, bitstream, Linux on KV260 |

## Notes

- Keep all paths free of spaces (Vivado/HLS break on them).
- Vitis 2025.2 is installed under `D:\Xilinx` (C: is nearly full). 2026.x moved Vivado to tiered paid licensing.
- Under Vitis, csim/cosim run the testbench's `--unit` / `--unit-small` IP tests: Vitis instruments every
  IP memory access, which makes the full network take hours. The full-network check runs natively (step 6).
