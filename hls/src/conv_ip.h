// int8 convolution IP for YOLO11n (HLS top function: conv_ip).
//
// One call computes one conv layer (any K, stride, padding, groups), CHW layout, batch 1:
//   acc[co][y][x] = bias[co] + sum (in - zp_in) * w           (int32, padding contributes 0)
//   raw_out = 1:  out32 = acc                                  (Detect logits)
//   raw_out = 0:  v = acc * mult[co]; v = silu ? SiLU(v) : v;  (mult = s_in * s_w[co])
//                 out8 = clamp(round_half_even(v * inv_s_out) + zp_out, -128, 127)
//
// Two implementations share this interface (pick one source file per build):
//   conv_ip_v0.cpp  baseline: 1 MAC/cycle, inputs read from DDR inside the MAC loop
//   conv_ip_v1.cpp  input row band + weights in BRAM, PO output channels in parallel
#pragma once
#include <cstdint>

// Buffer limits sized for YOLO11n at 640x640
constexpr int MAX_FMAP = 1638400;  // largest feature map (16 x 320 x 320 / 64 x 160 x 160)
constexpr int MAX_WGT = 294912;    // largest weight tensor (256 x 128 x 3 x 3)
constexpr int MAX_COUT = 384;
constexpr int MAX_WK = 2304;       // largest C_in/groups * K * K for one output channel

// v1 parameters
constexpr int PO = 16;             // output channels computed in parallel
constexpr int MAX_WO = 320;        // widest output row
constexpr int BANK_DEPTH = 2048;   // per-bank depth of the input row-band buffer (PO banks)

void conv_ip(const int8_t* in, const int8_t* wgt, const int32_t* bias, const float* mult,
             int8_t* out8, int32_t* out32,
             int c_in, int h, int w, int c_out, int k, int stride, int pad, int groups,
             int zp_in, float inv_s_out, int zp_out, bool silu, bool raw_out);

// True when a layer fits the v1 buffers: dense (groups = 1) or depthwise convs only.
inline bool conv_ip_v1_supports(int c_in, int w, int c_out, int k, int stride, int pad, int groups) {
    const bool dense = groups == 1, depthwise = groups == c_in && c_in == c_out;
    const int wp = w + 2 * pad, wo = (wp - k) / stride + 1;
    const int slots = dense ? (c_in + PO - 1) / PO : 1;  // channels stored per bank
    return (dense || depthwise) && slots * k * wp <= BANK_DEPTH && wo <= MAX_WO &&
           (dense ? c_in * k * k : k * k) <= MAX_WK && c_out <= MAX_COUT;
}
