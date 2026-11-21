#include "postprocess.h"

#include <algorithm>

static float iou(const Detection& a, const Detection& b) {
    float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0.f;
}

std::vector<Detection> nms(const Tensor& pred, float conf_thres, float iou_thres, int max_det) {
    // pred is stored as [rows = 4 + nc][cols = A] in a (c, h=1, w=A) or (rows, A, 1) layout: use raw indexing
    const int rows = pred.c, A = pred.h * pred.w, nc = rows - 4;
    std::vector<Detection> cand;
    for (int a = 0; a < A; a++) {
        int best = 0;
        float score = pred.d[size_t(4) * A + a];
        for (int k = 1; k < nc; k++) {
            float s = pred.d[size_t(4 + k) * A + a];
            if (s > score) score = s, best = k;
        }
        if (score <= conf_thres) continue;
        float cx = pred.d[a], cy = pred.d[size_t(A) + a], w = pred.d[size_t(2) * A + a], h = pred.d[size_t(3) * A + a];
        cand.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, best});
    }
    std::sort(cand.begin(), cand.end(), [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

    std::vector<Detection> keep;
    std::vector<bool> removed(cand.size(), false);
    for (size_t i = 0; i < cand.size() && int(keep.size()) < max_det; i++) {
        if (removed[i]) continue;
        keep.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); j++)
            if (!removed[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > iou_thres) removed[j] = true;
    }
    return keep;
}

void unletterbox(std::vector<Detection>& dets, float ratio, float pad_x, float pad_y, int orig_w, int orig_h) {
    for (auto& d : dets) {
        d.x1 = std::clamp((d.x1 - pad_x) / ratio, 0.f, float(orig_w));
        d.x2 = std::clamp((d.x2 - pad_x) / ratio, 0.f, float(orig_w));
        d.y1 = std::clamp((d.y1 - pad_y) / ratio, 0.f, float(orig_h));
        d.y2 = std::clamp((d.y2 - pad_y) / ratio, 0.f, float(orig_h));
    }
}
