// YOLO11n float reference model, driven by data/weights/weights.txt.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ops.h"

struct AttnParams {
    int num_heads, key_dim, head_dim;
    float scale;
};

struct Layer {
    int i;
    std::string type;
    std::vector<int> from;  // -1 = previous layer
};

class Yolo11n {
public:
    explicit Yolo11n(const std::string& weights_dir);

    // Runs the network on a 3xSxS letterboxed input. Returns every top-level layer output;
    // the last one is Detect: [4 + nc] x 1 x A (cx, cy, w, h in pixels, then sigmoid scores).
    std::vector<Tensor> forward(const Tensor& x) const;

    const std::vector<Layer>& layers() const { return layers_; }

    // Replaces the float conv2d for every conv (e.g. to dispatch to the HLS IP from the testbench).
    using ConvFn = std::function<Tensor(const Tensor&, const ConvW&)>;
    void set_conv_fn(ConvFn fn) { conv_fn_ = std::move(fn); }

private:
    ConvFn conv_fn_;
    std::map<std::string, ConvW> convs_;
    std::map<std::string, bool> add_;
    std::map<std::string, AttnParams> attn_;
    std::vector<Layer> layers_;
    int nc_ = 80, reg_max_ = 16;
    std::vector<int> strides_;

    bool has(const std::string& conv_name) const { return convs_.count(conv_name) > 0; }
    bool add(const std::string& module) const;
    // Ultralytics Conv/DWConv (prefix.conv) or a bare nn.Conv2d (prefix)
    Tensor conv(const Tensor& x, const std::string& prefix) const;

    Tensor bottleneck(const Tensor& x, const std::string& p) const;
    Tensor c3k(const Tensor& x, const std::string& p) const;
    Tensor c3k2(const Tensor& x, const std::string& p) const;
    Tensor sppf(const Tensor& x, const std::string& p) const;
    Tensor attention(const Tensor& x, const std::string& p) const;
    Tensor psablock(const Tensor& x, const std::string& p) const;
    Tensor c2psa(const Tensor& x, const std::string& p) const;
    Tensor detect(const std::vector<const Tensor*>& feats, const std::string& p) const;
};
