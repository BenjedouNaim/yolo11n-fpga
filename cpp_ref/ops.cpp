#include "ops.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

Tensor conv2d(const Tensor& x, const ConvW& cv) {
    if (x.c != cv.c_in) throw std::runtime_error(cv.name + ": input channels mismatch");
    const int ho = (x.h + 2 * cv.pad - cv.k) / cv.stride + 1;
    const int wo = (x.w + 2 * cv.pad - cv.k) / cv.stride + 1;
    const int cin_g = cv.c_in / cv.groups, cout_g = cv.c_out / cv.groups;
    Tensor y(cv.c_out, ho, wo);

    for (int co = 0; co < cv.c_out; co++) {
        const int g = co / cout_g;
        float* out = y.ch(co);
        std::fill(out, out + size_t(ho) * wo, cv.b.empty() ? 0.f : cv.b[co]);
        for (int ci = 0; ci < cin_g; ci++) {
            const float* in = x.ch(g * cin_g + ci);
            const float* wk = &cv.w[((size_t(co) * cin_g + ci) * cv.k) * cv.k];
            for (int ky = 0; ky < cv.k; ky++)
                for (int kx = 0; kx < cv.k; kx++) {
                    const float wv = wk[ky * cv.k + kx];
                    for (int oy = 0; oy < ho; oy++) {
                        const int iy = oy * cv.stride + ky - cv.pad;
                        if (iy < 0 || iy >= x.h) continue;
                        for (int ox = 0; ox < wo; ox++) {
                            const int ix = ox * cv.stride + kx - cv.pad;
                            if (ix >= 0 && ix < x.w) out[oy * wo + ox] += wv * in[iy * x.w + ix];
                        }
                    }
                }
        }
    }
    if (cv.act == Act::SiLU) silu_inplace(y);
    return y;
}

void silu_inplace(Tensor& x) {
    for (float& v : x.d) v = v / (1.f + std::exp(-v));
}

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor y = a;
    for (size_t i = 0; i < y.size(); i++) y.d[i] += b.d[i];
    return y;
}

Tensor concat(const std::vector<const Tensor*>& xs) {
    int c = 0;
    for (auto* t : xs) c += t->c;
    Tensor y(c, xs[0]->h, xs[0]->w);
    float* dst = y.d.data();
    for (auto* t : xs) dst = std::copy(t->d.begin(), t->d.end(), dst);
    return y;
}

Tensor slice_channels(const Tensor& x, int from, int n) {
    Tensor y(n, x.h, x.w);
    std::copy(x.ch(from), x.ch(from) + y.size(), y.d.begin());
    return y;
}

Tensor maxpool2d(const Tensor& x, int k, int stride, int pad) {
    const int ho = (x.h + 2 * pad - k) / stride + 1, wo = (x.w + 2 * pad - k) / stride + 1;
    Tensor y(x.c, ho, wo);
    for (int c = 0; c < x.c; c++)
        for (int oy = 0; oy < ho; oy++)
            for (int ox = 0; ox < wo; ox++) {
                float m = -FLT_MAX;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                        if (iy >= 0 && iy < x.h && ix >= 0 && ix < x.w) m = std::max(m, x.at(c, iy, ix));
                    }
                y.at(c, oy, ox) = m;
            }
    return y;
}

Tensor upsample_nearest2x(const Tensor& x) {
    Tensor y(x.c, x.h * 2, x.w * 2);
    for (int c = 0; c < x.c; c++)
        for (int yy = 0; yy < y.h; yy++)
            for (int xx = 0; xx < y.w; xx++) y.at(c, yy, xx) = x.at(c, yy / 2, xx / 2);
    return y;
}
