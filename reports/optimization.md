# HLS IP optimization study

Tool: Vitis HLS 2025.2 (`vitis-run`), part `xck26-sfvc784-2LV-c` (KV260), clock target 5 ns (200 MHz).
Reports: `conv_ip_v0_csynth.rpt`, `conv_ip_v1_csynth.rpt`, `conv_ip_v0_cosim.rpt`, `conv_ip_v1_cosim.rpt`.

## Verification

| Check | v0 | v1 |
| --- | --- | --- |
| Full-network C simulation (native, 174 IP calls, 2 images) | PASS, 0 mismatches | PASS, 0 mismatches |
| Detections vs PyTorch int8 | 8/8, IoU >= 0.97 | 8/8, IoU >= 0.97 |
| Vitis csim, unit test (4 layer types, YOLO11n shapes) | PASS, 0 mismatches | PASS, 0 mismatches |
| C/RTL co-simulation (XSIM, 4 reduced layers) | PASS, 0 mismatches | PASS, 0 mismatches |
| IP exported | `ip_export/conv_ip_v0.zip` | `ip_export/conv_ip_v1.zip` |

The unit test covers every IP mode: dense 3x3 + SiLU, depthwise 3x3, depthwise 3x3 + SiLU, 1x1 with int32 output.

## Results

| Version | What changed | Est. clock | BRAM_18K | DSP | FF | LUT | Co-sim cycles (4 test layers) | Speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0 | Baseline: 1 MAC/cycle, input read from DDR inside the MAC loop | 3.65 ns | 6 (2%) | 63 (5%) | 11,346 (4%) | 11,775 (10%) | 814,316 | 1x |
| v1 | Weights + pre-padded input row band in BRAM, 16 output channels in parallel, single epilogue unit | 3.65 ns | 68 (23%) | 79 (6%) | 17,442 (7%) | 18,227 (15%) | 71,762 | **11.3x** |

Both meet timing at 200 MHz with margin (about 270 MHz possible). v1 costs 62 more BRAMs (the
row-band, weight and accumulator buffers) for an 11x reduction in measured cycles.

Loop pipelining (from the synthesis log):

| Loop | v0 | v1 |
| --- | --- | --- |
| MAC loop | II = 1, depth 18 | II = 1, depth 9 |
| Weight load | II = 1 | II = 1 |
| Input row band load | — | II = 1 |
| Epilogue / write-back | (inside pixel loop) | II = 1, depth 77 |
| Bias / scale load | — | II = 2 (bias and scale share one AXI bundle; negligible: 16 per block) |

## Full-network projection (640x640, 3.24 GMAC per image)

| Version | Ideal cycle model | Model / co-sim ratio on test layers | Projected time per image at 200 MHz |
| --- | --- | --- | --- |
| v0 | 16.5 s | 3.4x | ~56 s |
| v1 | 1.83 s | 1.6x | ~2.9 s |

The ideal model (II = 1, no overheads) underestimates both. v0 loses about 28 cycles per output pixel
to pipeline fill and drain. v1 loses pipeline fill once per output row (MAC 9, load 16 and write 77
cycles). The projection scales the model by the ratio measured in co-simulation, so it is an estimate.
The co-simulation AXI model also has zero memory latency, so real DDR makes v0 slower still. Final
numbers come from on-board measurements in Phase 2.

## Where v1 still loses time, and v2

v1 reaches 16 MAC/cycle only inside the MAC loop. Per output row it also reloads the input row band
(`C_in x K x W` cycles; equal to the MAC time for 1x1 convs) and drains three pipelines. Next steps:

1. Overlap row-band loading, MACs and write-back (ping-pong buffers / `DATAFLOW`).
2. Sliding line buffer, so each input row is read once per block instead of K times.
3. Wider AXI ports (64-128 bit) to move 8-16 int8 values per cycle.
4. More parallelism (e.g. 32 output x 4 input channels). v1 uses only 6% of the 1248 DSPs.
