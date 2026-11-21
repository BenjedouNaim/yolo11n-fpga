// Testbench for conv_ip. It plays the processor's role in the HW/SW co-design:
// the YOLO11n graph, attention, DFL and NMS run here in software, and every conv is
// quantized to int8, sent to the IP, checked against an independent software int8 conv,
// and dequantized back. The final detections are compared with the PyTorch int8 model.
//
// Usage: tb <data_dir> [--cosim]
//   default   C simulation: every conv (87 per image) goes through the IP, all images
//   --cosim   RTL co-simulation: only a few small representative convs go through the IP
//             (RTL simulation is slow), first image only; the rest uses the software conv
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "../../cpp_ref/model.h"
#include "../../cpp_ref/postprocess.h"
#include "../src/conv_ip.h"

namespace fs = std::filesystem;

struct QConv {
    float s_in = 0, s_out = 0;
    int zp_in = 0, zp_out = 0;
    std::vector<int8_t> wq;    // int8 weights, same layout as the float ones
    std::vector<float> mult;   // s_in * s_w[c]
    std::vector<int32_t> bq;   // round(b / (s_in * s_w[c]))
    bool raw() const { return s_out == 0.f; }
};

struct Stats {
    int calls = 0, ip_calls = 0, mismatched_calls = 0;
    long long mismatches = 0;
    int max_diff = 0;
    double macs = 0, cycles_v0 = 0, cycles_v1 = 0;  // ideal-pipeline estimates, all IP calls
};

// Ideal cycle counts (II = 1 everywhere, no DDR stalls) from the loop structure of each version.
// They predict the trend; the real numbers come from C synthesis and co-simulation.
static double cycles_v0(const ConvW& cv, int ho, int wo) {
    const double wk = double(cv.c_in / cv.groups) * cv.k * cv.k;
    return cv.c_out * (wk + double(ho) * wo * (wk + 3));
}
static double cycles_v1(const ConvW& cv, int w, int ho, int wo) {
    const bool dw = cv.groups != 1;
    const double wk = dw ? cv.k * cv.k : double(cv.c_in) * cv.k * cv.k;
    const double band = double(cv.k) * (w + 2 * cv.pad), n_ci = dw ? PO : cv.c_in;
    const double blocks = (cv.c_out + PO - 1) / PO;
    return blocks * (PO * wk + PO + ho * (n_ci * band + wo * wk + double(PO) * wo));
}

static std::vector<int8_t> read_i8(const std::string& path, size_t expect) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int8_t> v(expect);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(expect));
    if (size_t(f.gcount()) != expect || f.peek() != EOF) throw std::runtime_error(path + ": unexpected size");
    return v;
}

static std::map<std::string, QConv> load_quant(const fs::path& qdir, const fs::path& wdir) {
    std::map<std::string, QConv> db;
    std::ifstream in(qdir / "qparams.txt");
    if (!in) throw std::runtime_error("cannot open " + (qdir / "qparams.txt").string());
    std::string kind, name;
    double s_in, s_out;
    int zp_in, zp_out;
    while (in >> kind >> name >> s_in >> zp_in >> s_out >> zp_out) {
        QConv q;
        q.s_in = float(s_in), q.zp_in = zp_in, q.s_out = float(s_out), q.zp_out = zp_out;
        db[name] = std::move(q);
    }
    (void)wdir;
    return db;
}

// Fills weights, multipliers and int32 bias the first time a conv is seen.
static void prepare(QConv& q, const ConvW& cv, const fs::path& qdir) {
    if (!q.wq.empty()) return;
    std::string key = cv.name;
    std::replace(key.begin(), key.end(), '.', '_');
    q.wq = read_i8((qdir / (key + ".wq.bin")).string(), cv.w.size());
    std::vector<float> sw = read_bin((qdir / (key + ".wscale.bin")).string(), size_t(cv.c_out));
    q.mult.resize(cv.c_out);
    q.bq.assign(cv.c_out, 0);
    for (int c = 0; c < cv.c_out; c++) {
        q.mult[c] = q.s_in * sw[c];
        if (!cv.b.empty()) q.bq[c] = int32_t(std::nearbyint(cv.b[c] / q.mult[c]));
    }
}

static std::vector<int8_t> quantize(const Tensor& x, float s, int zp) {
    const float inv = 1.0f / s;
    std::vector<int8_t> q(x.size());
    for (size_t i = 0; i < q.size(); i++) {
        float v = std::nearbyint(x.d[i] * inv) + float(zp);
        q[i] = int8_t(std::min(127.f, std::max(-128.f, v)));
    }
    return q;
}

// Independent software int8 conv (different loop order from the IP) used as the reference.
static void ref_conv(const std::vector<int8_t>& in, const ConvW& cv, const QConv& q, int h, int w,
                     std::vector<int8_t>& out8, std::vector<int32_t>& out32) {
    const int ho = (h + 2 * cv.pad - cv.k) / cv.stride + 1, wo = (w + 2 * cv.pad - cv.k) / cv.stride + 1;
    const int cin_g = cv.c_in / cv.groups, cout_g = cv.c_out / cv.groups;
    std::vector<int32_t> acc(size_t(ho) * wo);
    const float inv_s_out = q.raw() ? 0.f : 1.0f / q.s_out;
    for (int co = 0; co < cv.c_out; co++) {
        std::fill(acc.begin(), acc.end(), q.bq[co]);
        for (int ci = 0; ci < cin_g; ci++) {
            const int8_t* src = &in[size_t((co / cout_g) * cin_g + ci) * h * w];
            for (int ky = 0; ky < cv.k; ky++)
                for (int kx = 0; kx < cv.k; kx++) {
                    const int32_t wv = q.wq[((size_t(co) * cin_g + ci) * cv.k + ky) * cv.k + kx];
                    for (int oy = 0; oy < ho; oy++) {
                        const int iy = oy * cv.stride + ky - cv.pad;
                        if (iy < 0 || iy >= h) continue;
                        for (int ox = 0; ox < wo; ox++) {
                            const int ix = ox * cv.stride + kx - cv.pad;
                            if (ix >= 0 && ix < w) acc[size_t(oy) * wo + ox] += (int32_t(src[iy * w + ix]) - q.zp_in) * wv;
                        }
                    }
                }
        }
        for (size_t i = 0; i < acc.size(); i++) {
            const size_t o = size_t(co) * ho * wo + i;
            if (q.raw()) {
                out32[o] = acc[i];
                continue;
            }
            float v = float(acc[i]) * q.mult[co];
            if (cv.act == Act::SiLU) v = v / (1.0f + std::exp(-v));
            float r = std::nearbyint(v * inv_s_out) + float(q.zp_out);
            out8[o] = int8_t(std::min(127.f, std::max(-128.f, r)));
        }
    }
}

static Tensor dequantize(const QConv& q, const std::vector<int8_t>& out8, const std::vector<int32_t>& out32,
                         int c, int h, int w) {
    Tensor y(c, h, w);
    const size_t hw = size_t(h) * w;
    for (size_t i = 0; i < y.size(); i++)
        y.d[i] = q.raw() ? float(out32[i]) * q.mult[i / hw] : (float(out8[i]) - float(q.zp_out)) * q.s_out;
    return y;
}

static const float kConfTol = 0.05f, kIouMin = 0.90f;

static float box_iou(const Detection& a, const Detection& b) {
    float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float inter = iw * ih, uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0 ? inter / uni : 0.f;
}

static bool compare_detections(const Tensor& pred, const fs::path& gdir) {
    std::ifstream meta(gdir / "meta.txt");
    int oh, ow;
    float ratio, pad_x, pad_y, conf, iou;
    meta >> oh >> ow >> ratio >> pad_x >> pad_y >> conf >> iou;
    std::vector<Detection> dets = nms(pred, conf, iou);
    unletterbox(dets, ratio, pad_x, pad_y, ow, oh);

    std::vector<Detection> ref;
    std::ifstream gd(gdir / "detections_int8.txt");
    for (Detection d; gd >> d.cls >> d.conf >> d.x1 >> d.y1 >> d.x2 >> d.y2;) ref.push_back(d);

    // Two int8 implementations differ only by rounding ties (float vs exact int32 accumulation),
    // so detections must agree in class, closely in confidence, and overlap strongly.
    bool ok = dets.size() == ref.size();
    std::printf("  detections: IP flow %zu, PyTorch int8 %zu  (pass: same class, |dconf| < %.2f, IoU >= %.2f)\n",
                dets.size(), ref.size(), kConfTol, kIouMin);
    for (size_t i = 0; i < std::min(dets.size(), ref.size()); i++) {
        const Detection &a = dets[i], &b = ref[i];
        const float overlap = box_iou(a, b);
        bool pass = a.cls == b.cls && std::fabs(a.conf - b.conf) < kConfTol && overlap >= kIouMin;
        ok &= pass;
        std::printf("    cls %2d conf %.4f [%7.1f %7.1f %7.1f %7.1f]  vs ref cls %2d conf %.4f  IoU %.3f  %s\n",
                    a.cls, a.conf, a.x1, a.y1, a.x2, a.y2, b.cls, b.conf, overlap, pass ? "PASS" : "FAIL");
    }
    return ok;
}

static void compare_layers(const Yolo11n& net, const std::vector<Tensor>& outs, const fs::path& gdir) {
    std::printf("  %-26s %10s %10s %10s\n", "layer vs PyTorch int8", "max|err|", "mean|err|", "max|ref|");
    for (size_t i = 0; i + 1 < outs.size(); i++) {
        const Layer& l = net.layers()[i];
        char name[64];
        std::snprintf(name, sizeof name, "layer_%02d_%s_int8.bin", l.i, l.type.c_str());
        std::vector<float> ref = read_bin((gdir / name).string(), outs[i].size());
        double sum = 0;
        float mx = 0, mref = 0;
        for (size_t k = 0; k < ref.size(); k++) {
            float d = std::fabs(outs[i].d[k] - ref[k]);
            sum += d, mx = std::max(mx, d), mref = std::max(mref, std::fabs(ref[k]));
        }
        std::printf("  %-26s %10.4f %10.6f %10.3f\n", name, mx, sum / ref.size(), mref);
    }
}

// --unit: IP-only test on YOLO11n layer shapes with deterministic pseudo-random int8 data.
// No network forward pass, so it stays fast under Vitis csim/cosim instrumentation.
static const size_t kDepthFmap = 102400, kDepthWgt = 16384, kDepthCh = 384;  // = m_axi depth pragmas

// small = true: reduced sizes for RTL co-simulation (XSIM is slow and memory-hungry on long transactions)
static int run_unit_tests(bool small) {
    struct Case { const char* name; int c_in, hw, c_out, k, groups; bool silu, raw; };
    const Case full[] = {
        {"dense 3x3 + SiLU", 32, 20, 32, 3, 1, true, false},
        {"depthwise 3x3 (attn.pe)", 128, 20, 128, 3, 128, false, false},
        {"depthwise 3x3 + SiLU (cv3)", 256, 20, 256, 3, 256, true, false},
        {"1x1 int32 out (cv2 logits)", 64, 20, 64, 1, 1, false, true},
    };
    const Case reduced[] = {
        {"dense 3x3 + SiLU", 16, 8, 16, 3, 1, true, false},
        {"depthwise 3x3", 32, 8, 32, 3, 32, false, false},
        {"depthwise 3x3 + SiLU", 32, 8, 32, 3, 32, true, false},
        {"1x1 int32 out", 32, 8, 16, 1, 1, false, true},
    };
    std::vector<Case> cases = small ? std::vector<Case>(std::begin(reduced), std::end(reduced))
                                    : std::vector<Case>(std::begin(full), std::end(full));
    uint32_t seed = 12345;
    auto rnd = [&seed](int lo, int hi) { seed = seed * 1664525u + 1013904223u; return lo + int((seed >> 8) % uint32_t(hi - lo + 1)); };
    bool ok = true;
    for (const Case& c : cases) {
        ConvW cv;
        cv.name = c.name, cv.c_in = c.c_in, cv.c_out = c.c_out, cv.k = c.k, cv.stride = 1, cv.pad = c.k / 2;
        cv.groups = c.groups, cv.act = c.silu ? Act::SiLU : Act::None;
        const size_t wsize = size_t(c.c_out) * (c.c_in / c.groups) * c.k * c.k;
        QConv q;
        q.s_in = 0.05f, q.zp_in = -20, q.s_out = c.raw ? 0.f : 0.1f, q.zp_out = c.raw ? 0 : -100;
        q.wq.resize(wsize), q.mult.resize(c.c_out), q.bq.resize(c.c_out);
        for (auto& v : q.wq) v = int8_t(rnd(-127, 127));
        for (int i = 0; i < c.c_out; i++) q.mult[i] = q.s_in * 0.001f * float(rnd(1, 20)), q.bq[i] = rnd(-2000, 2000);
        std::vector<int8_t> in(size_t(c.c_in) * c.hw * c.hw);
        for (auto& v : in) v = int8_t(rnd(-128, 127));

        const size_t n_out = size_t(c.c_out) * c.hw * c.hw;
        std::vector<int8_t> ref8(n_out), ip8(n_out);
        std::vector<int32_t> ref32(n_out), ip32(n_out);
        ref_conv(in, cv, q, c.hw, c.hw, ref8, ref32);
        // Co-simulation copies `depth` elements from every m_axi pointer (see the IP's INTERFACE
        // pragmas), so the buffers handed to the IP must be at least that large.
        in.resize(std::max(in.size(), kDepthFmap));
        ip8.resize(std::max(n_out, kDepthFmap));
        ip32.resize(std::max(n_out, kDepthFmap));
        q.wq.resize(std::max(wsize, kDepthWgt));
        q.mult.resize(std::max<size_t>(c.c_out, kDepthCh));
        q.bq.resize(std::max<size_t>(c.c_out, kDepthCh));
        conv_ip(in.data(), q.wq.data(), q.bq.data(), q.mult.data(), ip8.data(), ip32.data(), c.c_in, c.hw, c.hw,
                c.c_out, c.k, 1, cv.pad, c.groups, q.zp_in, c.raw ? 0.f : 1.0f / q.s_out, q.zp_out, c.silu, c.raw);
        long long bad = 0;
        int maxd = 0;
        for (size_t i = 0; i < n_out; i++) {
            long long d = c.raw ? std::llabs((long long)ip32[i] - ref32[i]) : std::abs(int(ip8[i]) - int(ref8[i]));
            bad += d > 0;
            maxd = int(std::max<long long>(maxd, d));
        }
        const bool pass = maxd <= 1;  // RTL float exp may differ by 1 ulp -> at most 1 LSB
        ok &= pass;
        std::printf("  %-28s %4d->%-4d k%d g%-3d %dx%d  mismatches %lld  max diff %d  %s\n", c.name, c.c_in, c.c_out,
                    c.k, c.groups, c.hw, c.hw, bad, maxd, pass ? "PASS" : "FAIL");
    }
    std::printf("UNIT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // show progress live in tool logs
    if (argc > 1 && std::string(argv[argc - 1]) == "--unit") return run_unit_tests(false);
    if (argc > 1 && std::string(argv[argc - 1]) == "--unit-small") return run_unit_tests(true);
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: tb <data_dir> [--cosim]\n");
            return 2;
        }
        const fs::path data = argv[1];
        const bool cosim = argc > 2 && std::string(argv[2]) == "--cosim";
        // Small convs covering every IP mode: depthwise, depthwise + SiLU, 1x1 with int32 output
        const std::set<std::string> cosim_convs = {"model.10.m.0.attn.pe.conv", "model.23.cv3.2.0.0.conv",
                                                   "model.23.cv2.2.2"};
        const int tol = cosim ? 1 : 0;  // RTL float exp may differ by 1 ulp -> at most 1 LSB

        Yolo11n net((data / "weights").string());
        auto qdb = load_quant(data / "quant", data / "weights");
        Stats st;

        net.set_conv_fn([&](const Tensor& x, const ConvW& cv) {
            QConv& q = qdb.at(cv.name);
            prepare(q, cv, data / "quant");
            const int ho = (x.h + 2 * cv.pad - cv.k) / cv.stride + 1, wo = (x.w + 2 * cv.pad - cv.k) / cv.stride + 1;
            const size_t n_out = size_t(cv.c_out) * ho * wo;
            std::vector<int8_t> in = quantize(x, q.s_in, q.zp_in);
            std::vector<int8_t> ref8(n_out), ip8(n_out);
            std::vector<int32_t> ref32(n_out), ip32(n_out);
            ref_conv(in, cv, q, x.h, x.w, ref8, ref32);
            st.calls++;

            if (!cosim || cosim_convs.count(cv.name)) {
#ifdef CONV_IP_V1
                if (!conv_ip_v1_supports(cv.c_in, x.w, cv.c_out, cv.k, cv.stride, cv.pad, cv.groups))
                    throw std::runtime_error(cv.name + ": layer does not fit the v1 buffers");
#endif
                st.macs += double(n_out) * (cv.c_in / cv.groups) * cv.k * cv.k;
                st.cycles_v0 += cycles_v0(cv, ho, wo);
                st.cycles_v1 += cycles_v1(cv, x.w, ho, wo);
                conv_ip(in.data(), q.wq.data(), q.bq.data(), q.mult.data(), ip8.data(), ip32.data(),
                        cv.c_in, x.h, x.w, cv.c_out, cv.k, cv.stride, cv.pad, cv.groups,
                        q.zp_in, q.raw() ? 0.f : 1.0f / q.s_out, q.zp_out, cv.act == Act::SiLU, q.raw());
                st.ip_calls++;
                long long bad = 0;
                int maxd = 0;
                for (size_t i = 0; i < n_out; i++) {
                    long long d = q.raw() ? std::llabs((long long)ip32[i] - ref32[i]) : std::abs(int(ip8[i]) - int(ref8[i]));
                    if (d > 0) bad++;
                    maxd = int(std::max<long long>(maxd, d));
                }
                if (bad) st.mismatched_calls++;
                st.mismatches += bad;
                st.max_diff = std::max(st.max_diff, maxd);
                if (cosim || maxd > tol)
                    std::printf("  IP %-28s %4d->%-4d k%d s%d g%-3d %3dx%-3d  mismatches %lld  max diff %d\n", cv.name.c_str(),
                                cv.c_in, cv.c_out, cv.k, cv.stride, cv.groups, ho, wo, bad, maxd);
                return dequantize(q, ip8, ip32, cv.c_out, ho, wo);
            }
            return dequantize(q, ref8, ref32, cv.c_out, ho, wo);
        });

        bool all_ok = true;
        std::vector<fs::path> dirs;
        for (const auto& e : fs::directory_iterator(data / "golden"))
            if (e.is_directory()) dirs.push_back(e.path());
        std::sort(dirs.begin(), dirs.end());
        if (cosim) dirs.resize(1);

        for (const fs::path& gdir : dirs) {
            std::vector<float> in = read_bin((gdir / "input.bin").string());
            const int S = int(std::lround(std::sqrt(in.size() / 3.0)));
            Tensor x(3, S, S);
            x.d = std::move(in);
            std::printf("\n== %s (%dx%d) %s\n", gdir.filename().string().c_str(), S, S, cosim ? "[cosim]" : "[csim]");

            std::vector<Tensor> outs = net.forward(x);
            const Tensor& pred = outs.back();
            if (!cosim) compare_layers(net, outs, gdir);
            std::vector<float> ref = read_bin((gdir / "layer_23_Detect_0_int8.bin").string(), pred.size());
            const size_t A = size_t(pred.w);
            float box_err = 0.f, score_err = 0.f;
            for (size_t i = 0; i < ref.size(); i++) {
                float d = std::fabs(pred.d[i] - ref[i]);
                (i < 4 * A ? box_err : score_err) = std::max(i < 4 * A ? box_err : score_err, d);
            }
            std::printf("  Detect output vs PyTorch int8: max box err %.3f px, max score err %.4f\n", box_err, score_err);
            all_ok &= compare_detections(pred, gdir);
        }

        const bool ip_ok = st.max_diff <= tol;
        std::printf("\nIP vs software int8 conv: %d IP calls (%d convs total), %d with mismatches, %lld mismatched values, max diff %d -> %s\n",
                    st.ip_calls, st.calls, st.mismatched_calls, st.mismatches, st.max_diff, ip_ok ? "PASS" : "FAIL");
        all_ok &= ip_ok;
        const double n = double(dirs.size()), mhz = 200.0;
        std::printf("Per image: %.2f GMAC; ideal cycles v0 %.3g (%.0f ms), v1 %.3g (%.0f ms) at %.0f MHz -> %.1fx\n",
                    st.macs / n / 1e9, st.cycles_v0 / n, st.cycles_v0 / n / (mhz * 1e3), st.cycles_v1 / n,
                    st.cycles_v1 / n / (mhz * 1e3), mhz, st.cycles_v0 / st.cycles_v1);
        std::printf("OVERALL: %s\n", all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
