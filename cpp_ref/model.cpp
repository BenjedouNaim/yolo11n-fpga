#include "model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

Yolo11n::Yolo11n(const std::string& weights_dir) {
    std::ifstream in(weights_dir + "/weights.txt");
    if (!in) throw std::runtime_error("cannot open " + weights_dir + "/weights.txt");
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string kind;
        ss >> kind;
        if (kind == "conv") {
            ConvW cv;
            int silu, bias;
            ss >> cv.name >> cv.c_out >> cv.c_in >> cv.k >> cv.stride >> cv.pad >> cv.groups >> silu >> bias;
            cv.act = silu ? Act::SiLU : Act::None;
            std::string key = cv.name;
            std::replace(key.begin(), key.end(), '.', '_');
            cv.w = read_bin(weights_dir + "/" + key + ".weight.bin", size_t(cv.c_out) * (cv.c_in / cv.groups) * cv.k * cv.k);
            if (bias) cv.b = read_bin(weights_dir + "/" + key + ".bias.bin", cv.c_out);
            convs_[cv.name] = std::move(cv);
        } else if (kind == "flag") {
            std::string name, attr;
            int v;
            ss >> name >> attr >> v;
            if (attr == "add") add_[name] = v != 0;
        } else if (kind == "attn") {
            std::string name;
            AttnParams a;
            ss >> name >> a.num_heads >> a.key_dim >> a.head_dim >> a.scale;
            attn_[name] = a;
        } else if (kind == "detect") {
            ss >> nc_ >> reg_max_;
            for (int s; ss >> s;) strides_.push_back(s);
        } else if (kind == "layer") {
            Layer l;
            ss >> l.i >> l.type;
            for (int f; ss >> f;) l.from.push_back(f);
            layers_.push_back(l);
        }
    }
}

bool Yolo11n::add(const std::string& module) const {
    auto it = add_.find(module);
    return it != add_.end() && it->second;
}

Tensor Yolo11n::conv(const Tensor& x, const std::string& prefix) const {
    auto it = convs_.find(prefix + ".conv");
    if (it == convs_.end()) it = convs_.find(prefix);
    if (it == convs_.end()) throw std::runtime_error("missing conv " + prefix);
    return conv_fn_ ? conv_fn_(x, it->second) : conv2d(x, it->second);
}

Tensor Yolo11n::bottleneck(const Tensor& x, const std::string& p) const {
    Tensor y = conv(conv(x, p + ".cv1"), p + ".cv2");
    return add(p) ? ::add(x, y) : y;
}

Tensor Yolo11n::c3k(const Tensor& x, const std::string& p) const {
    Tensor a = conv(x, p + ".cv1");
    for (int i = 0; has(p + ".m." + std::to_string(i) + ".cv1.conv"); i++) a = bottleneck(a, p + ".m." + std::to_string(i));
    Tensor b = conv(x, p + ".cv2");
    return conv(concat({&a, &b}), p + ".cv3");
}

Tensor Yolo11n::c3k2(const Tensor& x, const std::string& p) const {
    Tensor t = conv(x, p + ".cv1");
    const int c = t.c / 2;
    std::vector<Tensor> ys = {slice_channels(t, 0, c), slice_channels(t, c, c)};
    for (int i = 0; has(p + ".m." + std::to_string(i) + ".cv1.conv"); i++) {
        const std::string m = p + ".m." + std::to_string(i);
        ys.push_back(has(m + ".cv3.conv") ? c3k(ys.back(), m) : bottleneck(ys.back(), m));
    }
    std::vector<const Tensor*> refs;
    for (auto& y : ys) refs.push_back(&y);
    return conv(concat(refs), p + ".cv2");
}

Tensor Yolo11n::sppf(const Tensor& x, const std::string& p) const {
    std::vector<Tensor> ys = {conv(x, p + ".cv1")};
    for (int i = 0; i < 3; i++) ys.push_back(maxpool2d(ys.back(), 5, 1, 2));
    Tensor y = conv(concat({&ys[0], &ys[1], &ys[2], &ys[3]}), p + ".cv2");
    return add(p) ? ::add(x, y) : y;
}

Tensor Yolo11n::attention(const Tensor& x, const std::string& p) const {
    const AttnParams& a = attn_.at(p);
    const int N = x.h * x.w, per_head = 2 * a.key_dim + a.head_dim;
    Tensor qkv = conv(x, p + ".qkv");  // [num_heads * (q | k | v)] x H x W
    Tensor out(a.num_heads * a.head_dim, x.h, x.w), v_img(a.num_heads * a.head_dim, x.h, x.w);
    std::vector<float> attn(size_t(N) * N);

    for (int h = 0; h < a.num_heads; h++) {
        const float* q = qkv.ch(h * per_head);
        const float* k = qkv.ch(h * per_head + a.key_dim);
        const float* v = qkv.ch(h * per_head + 2 * a.key_dim);
        // attn[i][j] = softmax_j(scale * q_i . k_j)
        for (int i = 0; i < N; i++) {
            float* row = &attn[size_t(i) * N];
            float mx = -INFINITY;
            for (int j = 0; j < N; j++) {
                float s = 0.f;
                for (int d = 0; d < a.key_dim; d++) s += q[size_t(d) * N + i] * a.scale * k[size_t(d) * N + j];
                row[j] = s;
                mx = std::max(mx, s);
            }
            float sum = 0.f;
            for (int j = 0; j < N; j++) sum += (row[j] = std::exp(row[j] - mx));
            for (int j = 0; j < N; j++) row[j] /= sum;
        }
        // out[d][i] = sum_j v[d][j] * attn[i][j]
        for (int d = 0; d < a.head_dim; d++) {
            const float* vd = v + size_t(d) * N;
            float* od = out.ch(h * a.head_dim + d);
            for (int i = 0; i < N; i++) {
                const float* row = &attn[size_t(i) * N];
                float s = 0.f;
                for (int j = 0; j < N; j++) s += vd[j] * row[j];
                od[i] = s;
            }
            std::copy(vd, vd + N, v_img.ch(h * a.head_dim + d));
        }
    }
    Tensor pe = conv(v_img, p + ".pe");
    return conv(::add(out, pe), p + ".proj");
}

Tensor Yolo11n::psablock(const Tensor& x, const std::string& p) const {
    const bool sc = add(p);
    Tensor a = attention(x, p + ".attn");
    Tensor y = sc ? ::add(x, a) : a;
    Tensor f = conv(conv(y, p + ".ffn.0"), p + ".ffn.1");
    return sc ? ::add(y, f) : f;
}

Tensor Yolo11n::c2psa(const Tensor& x, const std::string& p) const {
    Tensor t = conv(x, p + ".cv1");
    const int c = t.c / 2;
    Tensor a = slice_channels(t, 0, c), b = slice_channels(t, c, c);
    for (int i = 0; has(p + ".m." + std::to_string(i) + ".attn.qkv.conv"); i++) b = psablock(b, p + ".m." + std::to_string(i));
    return conv(concat({&a, &b}), p + ".cv2");
}

Tensor Yolo11n::detect(const std::vector<const Tensor*>& feats, const std::string& p) const {
    int A = 0;
    for (auto* f : feats) A += f->h * f->w;
    Tensor y(4 + nc_, 1, A);
    int base = 0;
    for (size_t l = 0; l < feats.size(); l++) {
        const Tensor& f = *feats[l];
        const std::string i = std::to_string(l);
        Tensor box = conv(conv(conv(f, p + ".cv2." + i + ".0"), p + ".cv2." + i + ".1"), p + ".cv2." + i + ".2");
        Tensor cls = conv(conv(f, p + ".cv3." + i + ".0.0"), p + ".cv3." + i + ".0.1");
        cls = conv(conv(cls, p + ".cv3." + i + ".1.0"), p + ".cv3." + i + ".1.1");
        cls = conv(cls, p + ".cv3." + i + ".2");

        const float stride = float(strides_[l]);
        const int HW = f.h * f.w;
        for (int a = 0; a < HW; a++) {
            // DFL: expected value of a softmax over reg_max bins, per side (l, t, r, b)
            float dist[4];
            for (int s = 0; s < 4; s++) {
                float mx = -INFINITY, sum = 0.f, ev = 0.f;
                for (int b = 0; b < reg_max_; b++) mx = std::max(mx, box.d[size_t(s * reg_max_ + b) * HW + a]);
                for (int b = 0; b < reg_max_; b++) {
                    float e = std::exp(box.d[size_t(s * reg_max_ + b) * HW + a] - mx);
                    sum += e;
                    ev += e * b;
                }
                dist[s] = ev / sum;
            }
            const float ax = (a % f.w) + 0.5f, ay = (a / f.w) + 0.5f;
            const float x1 = ax - dist[0], y1 = ay - dist[1], x2 = ax + dist[2], y2 = ay + dist[3];
            const int col = base + a;
            y.d[col] = (x1 + x2) / 2 * stride;
            y.d[size_t(A) + col] = (y1 + y2) / 2 * stride;
            y.d[size_t(2) * A + col] = (x2 - x1) * stride;
            y.d[size_t(3) * A + col] = (y2 - y1) * stride;
            for (int k = 0; k < nc_; k++) y.d[size_t(4 + k) * A + col] = 1.f / (1.f + std::exp(-cls.d[size_t(k) * HW + a]));
        }
        base += HW;
    }
    return y;
}

std::vector<Tensor> Yolo11n::forward(const Tensor& x) const {
    std::vector<Tensor> outs;
    outs.reserve(layers_.size());
    for (const Layer& l : layers_) {
        auto input = [&](int f) -> const Tensor& { return f == -1 ? (l.i == 0 ? x : outs[l.i - 1]) : outs[f]; };
        const Tensor& in = input(l.from[0]);
        const std::string p = "model." + std::to_string(l.i);

        if (l.type == "Conv") outs.push_back(conv(in, p));
        else if (l.type == "C3k2") outs.push_back(c3k2(in, p));
        else if (l.type == "SPPF") outs.push_back(sppf(in, p));
        else if (l.type == "C2PSA") outs.push_back(c2psa(in, p));
        else if (l.type == "Upsample") outs.push_back(upsample_nearest2x(in));
        else if (l.type == "Concat" || l.type == "Detect") {
            std::vector<const Tensor*> xs;
            for (int f : l.from) xs.push_back(&input(f));
            outs.push_back(l.type == "Concat" ? concat(xs) : detect(xs, p));
        } else throw std::runtime_error("unsupported layer type " + l.type);
    }
    return outs;
}
