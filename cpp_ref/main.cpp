// Runs the C++ reference model on every exported image and compares it layer by layer
// with the PyTorch golden outputs, then compares the final detections.
//
// Usage: yolo_ref [data_dir]   (default: ../data relative to the working directory)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "model.h"
#include "postprocess.h"

namespace fs = std::filesystem;

static const float kRelTol = 1e-3f;  // max |err| / max |golden| per layer

static bool compare_layers(const Yolo11n& net, const std::vector<Tensor>& outs, const fs::path& gdir) {
    bool ok = true;
    std::printf("  %-22s %-16s %12s %12s  %s\n", "layer", "shape", "max|err|", "max|ref|", "result");
    for (size_t i = 0; i < outs.size(); i++) {
        const Layer& l = net.layers()[i];
        char name[64];
        std::snprintf(name, sizeof name, "layer_%02d_%s%s.bin", l.i, l.type.c_str(), l.type == "Detect" ? "_0" : "");
        const Tensor& o = outs[i];
        std::vector<float> ref = read_bin((gdir / name).string(), o.size());
        float max_err = 0.f, max_ref = 0.f;
        for (size_t k = 0; k < ref.size(); k++) {
            max_err = std::max(max_err, std::fabs(o.d[k] - ref[k]));
            max_ref = std::max(max_ref, std::fabs(ref[k]));
        }
        const bool pass = max_err <= kRelTol * std::max(1.f, max_ref);
        ok &= pass;
        char shape[32];
        std::snprintf(shape, sizeof shape, "%dx%dx%d", o.c, o.h, o.w);
        std::printf("  %-22s %-16s %12.3e %12.3e  %s\n", name, shape, max_err, max_ref, pass ? "PASS" : "FAIL");
    }
    return ok;
}

static bool compare_detections(std::vector<Detection> dets, const fs::path& gdir) {
    std::ifstream meta(gdir / "meta.txt");
    int oh, ow;
    float ratio, pad_x, pad_y, conf, iou;
    meta >> oh >> ow >> ratio >> pad_x >> pad_y >> conf >> iou;
    unletterbox(dets, ratio, pad_x, pad_y, ow, oh);

    std::vector<Detection> ref;
    std::ifstream gd(gdir / "detections.txt");
    for (Detection d; gd >> d.cls >> d.conf >> d.x1 >> d.y1 >> d.x2 >> d.y2;) ref.push_back(d);

    bool ok = dets.size() == ref.size();
    std::printf("  detections: C++ %zu, PyTorch %zu\n", dets.size(), ref.size());
    for (size_t i = 0; i < std::min(dets.size(), ref.size()); i++) {
        const Detection &a = dets[i], &b = ref[i];
        float box_err = std::max({std::fabs(a.x1 - b.x1), std::fabs(a.y1 - b.y1), std::fabs(a.x2 - b.x2), std::fabs(a.y2 - b.y2)});
        bool pass = a.cls == b.cls && std::fabs(a.conf - b.conf) < 1e-3f && box_err < 0.5f;
        ok &= pass;
        std::printf("    cls %2d conf %.4f [%7.1f %7.1f %7.1f %7.1f]  vs ref cls %2d conf %.4f  box err %.2f px  %s\n",
                    a.cls, a.conf, a.x1, a.y1, a.x2, a.y2, b.cls, b.conf, box_err, pass ? "PASS" : "FAIL");
    }
    return ok;
}

int main(int argc, char** argv) {
    try {
        const fs::path data = argc > 1 ? fs::path(argv[1]) : fs::path("../data");
        Yolo11n net((data / "weights").string());
        bool all_ok = true;

        for (const auto& entry : fs::directory_iterator(data / "golden")) {
            if (!entry.is_directory()) continue;
            const fs::path gdir = entry.path();
            std::vector<float> in = read_bin((gdir / "input.bin").string());
            const int S = int(std::lround(std::sqrt(in.size() / 3.0)));
            Tensor x(3, S, S);
            x.d = std::move(in);

            std::printf("\n== %s (%dx%d)\n", gdir.filename().string().c_str(), S, S);
            auto t0 = std::chrono::steady_clock::now();
            std::vector<Tensor> outs = net.forward(x);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  forward: %.0f ms\n", ms);

            bool ok = compare_layers(net, outs, gdir);
            std::ifstream meta(gdir / "meta.txt");
            float skip, conf, iou;
            for (int k = 0; k < 5; k++) meta >> skip;
            meta >> conf >> iou;
            ok &= compare_detections(nms(outs.back(), conf, iou), gdir);
            std::printf("  => %s\n", ok ? "PASS" : "FAIL");
            all_ok &= ok;
        }
        std::printf("\nOVERALL: %s\n", all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
