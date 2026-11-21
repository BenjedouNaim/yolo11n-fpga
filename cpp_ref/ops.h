// Primitive layers used by YOLO11n (float reference, no optimisation).
#pragma once
#include <string>
#include <vector>

#include "tensor.h"

enum class Act { None, SiLU };

// Conv2d with BatchNorm already fused into weight/bias (done by export_golden.py).
struct ConvW {
    std::string name;
    int c_out = 0, c_in = 0, k = 1, stride = 1, pad = 0, groups = 1;  // c_in = total input channels
    Act act = Act::None;
    std::vector<float> w;  // [c_out][c_in/groups][k][k]
    std::vector<float> b;  // [c_out], zeros if the conv has no bias
};

Tensor conv2d(const Tensor& x, const ConvW& cv);
void silu_inplace(Tensor& x);
Tensor add(const Tensor& a, const Tensor& b);
Tensor concat(const std::vector<const Tensor*>& xs);
// Channels [from, from + n)
Tensor slice_channels(const Tensor& x, int from, int n);
Tensor maxpool2d(const Tensor& x, int k, int stride, int pad);
Tensor upsample_nearest2x(const Tensor& x);
