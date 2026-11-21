// Post-processing on the processor side: NMS and mapping boxes back to the image.
#pragma once
#include <vector>

#include "tensor.h"

struct Detection {
    float x1, y1, x2, y2, conf;
    int cls;
};

// pred: [4 + nc, A] from Detect (cx, cy, w, h in letterbox pixels, then class scores).
// Class-aware NMS, mirroring ultralytics non_max_suppression defaults (max_det = 300).
std::vector<Detection> nms(const Tensor& pred, float conf_thres, float iou_thres, int max_det = 300);

// Undo the letterbox: subtract padding, divide by ratio, clip to the original image.
void unletterbox(std::vector<Detection>& dets, float ratio, float pad_x, float pad_y, int orig_w, int orig_h);
