// v1: on-chip buffering + output-channel parallelism.
//   * weights, bias and scale of PO output channels are loaded once per block into BRAM
//   * for each output row, the input row band (C x K rows x padded width) is copied into BRAM,
//     already padded with zp_in, so the MAC loop has no bounds checks and no DDR accesses
//   * PO output channels are computed in parallel (PO MACs per cycle):
//       dense:     one input value per cycle, broadcast to the PO lanes
//       depthwise: lane p reads its own channel from bank p
//   * the epilogue (rescale, SiLU, requantize) runs once per output value, one per cycle,
//     so only one float exp unit is instantiated
// Supports groups = 1 and depthwise convs (all YOLO11n layers); see conv_ip_v1_supports().
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

    const int wp = w + 2 * pad;
    const int ho = (h + 2 * pad - k) / stride + 1, wo = (wp - k) / stride + 1;
    const bool dw = groups != 1;  // depthwise: groups == c_in == c_out
    const int wk = dw ? k * k : c_in * k * k;
    const int band = k * wp;  // one channel's rows for one output row

    int8_t wbuf[PO][MAX_WK];
    int8_t rowbuf[PO][BANK_DEPTH];  // channel c -> bank c % PO, slot c / PO
    int32_t accbuf[PO][MAX_WO];
    int32_t bbuf[PO];
    float mbuf[PO];
#pragma HLS ARRAY_PARTITION variable=wbuf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=rowbuf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accbuf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bbuf complete
#pragma HLS ARRAY_PARTITION variable=mbuf complete

blk_loop:
    for (int co0 = 0; co0 < c_out; co0 += PO) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=24 avg=8
    load_w:
        for (int p = 0; p < PO; p++) {
            for (int i = 0; i < wk; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=9 max=2304 avg=576
                wbuf[p][i] = co0 + p < c_out ? wgt[(co0 + p) * wk + i] : int8_t(0);
            }
        }
    load_b:
        for (int p = 0; p < PO; p++) {
#pragma HLS PIPELINE II=1
            bbuf[p] = co0 + p < c_out ? bias[co0 + p] : 0;
            mbuf[p] = co0 + p < c_out ? mult[co0 + p] : 0.0f;
        }

        const int ci_lo = dw ? co0 : 0;
        const int n_ci = dw ? PO : c_in;

    row_loop:
        for (int oy = 0; oy < ho; oy++) {
#pragma HLS LOOP_TRIPCOUNT min=20 max=320 avg=80
            // 1. input row band -> BRAM, padding filled with zp_in (so it contributes 0)
            {
                int c = 0, ky = 0, xp = 0, bank = 0, slot = 0;
            load_rows:
                for (int t = 0; t < n_ci * band; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=60 max=33000 avg=8000
                    const int iy = oy * stride + ky - pad, ix = xp - pad, ch = ci_lo + c;
                    int8_t v = int8_t(zp_in);
                    if (ch < c_in && iy >= 0 && iy < h && ix >= 0 && ix < w) v = in[(ch * h + iy) * w + ix];
                    rowbuf[bank][slot + ky * wp + xp] = v;
                    if (++xp == wp) {
                        xp = 0;
                        if (++ky == k) {
                            ky = 0;
                            c++;
                            if (++bank == PO) {
                                bank = 0;
                                slot += band;
                            }
                        }
                    }
                }
            }

            // 2. MACs for PO output channels in parallel, flattened over (ox, i)
            {
                int32_t acc[PO];
#pragma HLS ARRAY_PARTITION variable=acc complete
                int ox = 0, i = 0, ky = 0, kx = 0, bank = 0, slot = 0;
            mac_loop:
                for (int t = 0; t < wo * wk; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=180 max=200000 avg=40000
                    const int col = ky * wp + ox * stride + kx;
                    const int32_t xb = int32_t(rowbuf[bank][slot + col]) - zp_in;  // dense: broadcast
                lanes:
                    for (int p = 0; p < PO; p++) {
#pragma HLS UNROLL
                        const int32_t x = dw ? int32_t(rowbuf[p][col]) - zp_in : xb;
                        acc[p] = (i == 0 ? bbuf[p] : acc[p]) + x * int32_t(wbuf[p][i]);
                        if (i == wk - 1) accbuf[p][ox] = acc[p];
                    }
                    // advance (ci, ky, kx) then ox
                    if (++i == wk) {
                        i = ky = kx = bank = slot = 0;
                        ox++;
                    } else if (++kx == k) {
                        kx = 0;
                        if (++ky == k) {
                            ky = 0;
                            if (++bank == PO) {
                                bank = 0;
                                slot += band;
                            }
                        }
                    }
                }
            }

            // 3. epilogue + write-back, one output value per cycle
        write_out:
            for (int p = 0; p < PO; p++) {
                for (int ox = 0; ox < wo; ox++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=20 max=320 avg=80
                    if (co0 + p < c_out) {
                        const int o = ((co0 + p) * ho + oy) * wo + ox;
                        if (raw_out)
                            out32[o] = accbuf[p][ox];
                        else
                            out8[o] = requant(accbuf[p][ox], mbuf[p], inv_s_out, zp_out, silu);
                    }
                }
            }
        }
    }
}
