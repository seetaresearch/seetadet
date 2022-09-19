/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_BBOX_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_BBOX_H_

#include "../../utils/detection/types.h"

#if defined(__CUDACC__)
#define HOSTDEVICE_DECL inline __host__ __device__
#else
#define HOSTDEVICE_DECL inline
#endif

namespace dragon {

namespace detection {

/*
 * BBox Functions.
 */

template <typename T, class BoxT>
inline void SortBoxes(const int N, T* data, bool descend = true) {
  auto* boxes = reinterpret_cast<BoxT*>(data);
  std::sort(boxes, boxes + N, [descend](BoxT lhs, BoxT rhs) {
    return descend ? (lhs.score > rhs.score) : (lhs.score < rhs.score);
  });
}

/*
 * BBox Utilities.
 */

namespace utils {

template <typename T>
HOSTDEVICE_DECL bool CheckIoU(const T thresh, const T* a, const T* b) {
#if defined(__CUDACC__)
  const T x1 = max(a[0], b[0]);
  const T y1 = max(a[1], b[1]);
  const T x2 = min(a[2], b[2]);
  const T y2 = min(a[3], b[3]);
  const T width = max(T(0), x2 - x1);
  const T height = max(T(0), y2 - y1);
#else
  const T x1 = std::max(a[0], b[0]);
  const T y1 = std::max(a[1], b[1]);
  const T x2 = std::min(a[2], b[2]);
  const T y2 = std::min(a[3], b[3]);
  const T width = std::max(T(0), x2 - x1);
  const T height = std::max(T(0), y2 - y1);
#endif
  const T inter = width * height;
  const T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  const T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return inter >= thresh * (Sa + Sb - inter);
}

template <typename T>
inline void BBoxTransform(
    const T dx,
    const T dy,
    const T dw,
    const T dh,
    const T im_w,
    const T im_h,
    const T im_scale_h,
    const T im_scale_w,
    T* bbox) {
  const T w = bbox[2] - bbox[0];
  const T h = bbox[3] - bbox[1];
  const T ctr_x = bbox[0] + T(0.5) * w;
  const T ctr_y = bbox[1] + T(0.5) * h;
  const T pred_ctr_x = dx * w + ctr_x;
  const T pred_ctr_y = dy * h + ctr_y;
  const T pred_w = std::exp(dw) * w;
  const T pred_h = std::exp(dh) * h;
  const T x1 = pred_ctr_x - T(0.5) * pred_w;
  const T y1 = pred_ctr_y - T(0.5) * pred_h;
  const T x2 = pred_ctr_x + T(0.5) * pred_w;
  const T y2 = pred_ctr_y + T(0.5) * pred_h;
  bbox[0] = std::max(T(0), std::min(x1, im_w)) / im_scale_w;
  bbox[1] = std::max(T(0), std::min(y1, im_h)) / im_scale_h;
  bbox[2] = std::max(T(0), std::min(x2, im_w)) / im_scale_w;
  bbox[3] = std::max(T(0), std::min(y2, im_h)) / im_scale_h;
}

template <typename T>
inline int GetBBoxLevel(
    const int lvl_min,
    const int lvl_max,
    const int lvl0,
    const int s0,
    T* bbox) {
  const T w = bbox[2] - bbox[0];
  const T h = bbox[3] - bbox[1];
  if (w <= T(0) || h <= T(0)) return -1;
  const T s = std::sqrt(w * h);
  const int lvl = lvl0 + std::log2(s / s0 + T(1e-6));
  return std::min(std::max(lvl, lvl_min), lvl_max);
}

} // namespace utils

} // namespace detection

} // namespace dragon

#undef HOSTDEVICE_DECL

#endif // DRAGON_EXTENSION_UTILS_DETECTION_BBOX_H_
