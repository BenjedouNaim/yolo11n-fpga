// v0 baseline: functionally complete, one MAC per cycle, inputs read straight from DDR.
// Optimisations (input tiling in BRAM, output-channel parallelism, wider AXI) come next.
#include "conv_ip.h"

#include <cmath>

static int8_t requant(int32_t acc, float mult, float inv_s_out, int zp_out, bool silu) {
#pragma HLS INLINE
    float v = float(acc) * mult;
    if (silu) v = v / (1.0f + std::exp(-v));
    float q = std::nearbyint(v * inv_s_out) + float(zp_out);
    q = q < -128.0f ? -128.0f : (q > 127.0f ? 127.0f : q);
    return int8_t(q);
}

void conv_ip(const int8_t* in, const int8_t* wgt, const int32_t* bias, const float* mult,
             int8_t* out8, int32_t* out32,
             int c_in, int h, int w, int c_out, int k, int stride, int pad, int groups,
             int zp_in, float inv_s_out, int zp_out, bool silu, bool raw_out) {
#pragma HLS INTERFACE m_axi port=in    offset=slave bundle=gmem0 depth=102400
#pragma HLS INTERFACE m_axi port=wgt   offset=slave bundle=gmem1 depth=16384
#pragma HLS INTERFACE m_axi port=bias  offset=slave bundle=gmem1 depth=384
#pragma HLS INTERFACE m_axi port=mult  offset=slave bundle=gmem1 depth=384
#pragma HLS INTERFACE m_axi port=out8  offset=slave bundle=gmem2 depth=102400
#pragma HLS INTERFACE m_axi port=out32 offset=slave bundle=gmem2 depth=102400
#pragma HLS INTERFACE s_axilite port=c_in
#pragma HLS INTERFACE s_axilite port=h
#pragma HLS INTERFACE s_axilite port=w
#pragma HLS INTERFACE s_axilite port=c_out
#pragma HLS INTERFACE s_axilite port=k
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=pad
#pragma HLS INTERFACE s_axilite port=groups
#pragma HLS INTERFACE s_axilite port=zp_in
#pragma HLS INTERFACE s_axilite port=inv_s_out
#pragma HLS INTERFACE s_axilite port=zp_out
#pragma HLS INTERFACE s_axilite port=silu
#pragma HLS INTERFACE s_axilite port=raw_out
#pragma HLS INTERFACE s_axilite port=return

    const int ho = (h + 2 * pad - k) / stride + 1;
    const int wo = (w + 2 * pad - k) / stride + 1;
    const int cin_g = c_in / groups, cout_g = c_out / groups;
    const int wk = cin_g * k * k;

    int8_t wbuf[MAX_WK];

co_loop:
    for (int co = 0; co < c_out; co++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=384 avg=128
    load_w:
        for (int i = 0; i < wk; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=9 max=2304 avg=576
            wbuf[i] = wgt[co * wk + i];
        }
        const int32_t b = bias[co];
        const float m = mult[co];
        const int ci0 = (co / cout_g) * cin_g;

    oy_loop:
        for (int oy = 0; oy < ho; oy++) {
#pragma HLS LOOP_TRIPCOUNT min=20 max=320 avg=80
        ox_loop:
            for (int ox = 0; ox < wo; ox++) {
#pragma HLS LOOP_TRIPCOUNT min=20 max=320 avg=80
                int32_t acc = b;
                int ci = 0, ky = 0, kx = 0;
            mac_loop:  // flattened ci/ky/kx with running counters (no div/mod)
                for (int i = 0; i < wk; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=9 max=2304 avg=576
                    const int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                    if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                        acc += (int32_t(in[((ci0 + ci) * h + iy) * w + ix]) - zp_in) * int32_t(wbuf[i]);
                    if (++kx == k) {
                        kx = 0;
                        if (++ky == k) {
                            ky = 0;
                            ci++;
                        }
                    }
                }
                const int o = (co * ho + oy) * wo + ox;
                if (raw_out)
                    out32[o] = acc;
                else
                    out8[o] = requant(acc, m, inv_s_out, zp_out, silu);
            }
        }
    }
}
