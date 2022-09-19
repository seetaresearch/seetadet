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

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_ANCHORS_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_ANCHORS_H_

#include "../../utils/detection/types.h"

namespace dragon {

namespace detection {

/*!
 * Anchor Functions.
 */

template <typename IndexT>
inline void SetGridArgs(
    const int num_anchors,
    const int num_cell_anchors,
    const IndexT* grid_info,
    vector<GridArgs<IndexT>>& grid_args) {
  IndexT grid_offset = 0;
  for (int i = 0; i < grid_args.size(); ++i, grid_info += 2) {
    auto& args = grid_args[i];
    args.h = grid_info[0];
    args.w = grid_info[1];
    args.size = num_cell_anchors * args.h * args.w;
    args.offset = grid_offset;
    grid_offset += args.size;
  }
  std::stringstream ss;
  if (grid_offset != num_anchors) {
    ss << "Mismatched number of anchors. (Excepted ";
    ss << num_anchors << ", Got " << grid_offset << ")";
    for (int i = 0; i < grid_args.size(); ++i) {
      ss << "\nGrid #" << i << ": "
         << "A=" << num_cell_anchors << ", H=" << grid_args[i].h
         << ", W=" << grid_args[i].w << "\n";
    }
  }
  if (!ss.str().empty()) LOG(FATAL) << ss.str();
}

template <typename T>
inline void GenerateAnchors(
    const int stride,
    const int num_ratios,
    const int num_scales,
    const T* ratios,
    const T* scales,
    T* anchors) {
  T* offset_anchors = anchors;
  T x = T(0.5) * T(stride), y = T(0.5) * T(stride);
  for (int i = 0; i < num_ratios; ++i) {
    const T ratio_w = std::sqrt(T(1) / ratios[i]);
    const T ratio_h = ratio_w * ratios[i];
    for (int j = 0; j < num_scales; ++j) {
      offset_anchors[0] = -x * ratio_w * scales[j];
      offset_anchors[1] = -y * ratio_h * scales[j];
      offset_anchors[2] = x * ratio_w * scales[j];
      offset_anchors[3] = y * ratio_h * scales[j];
      offset_anchors += 4;
    }
  }
}

template <typename T>
inline void GetAnchors(
    const int num_anchors,
    const int num_cell_anchors,
    const GridArgs<int64_t>& args,
    const T* cell_anchors,
    const int64_t* indices,
    T* anchors) {
  for (int i = 0; i < num_anchors; ++i) {
    auto index = indices[i];
    const auto w = index % args.w;
    index /= args.w;
    const auto h = index % args.h;
    index /= args.h;
    const auto shift_x = T(w * args.stride);
    const auto shift_y = T(h * args.stride);
    auto* offset_anchors = anchors + i * 5;
    const auto* offset_cell_anchors = cell_anchors + index * 4;
    offset_anchors[0] = shift_x + offset_cell_anchors[0];
    offset_anchors[1] = shift_y + offset_cell_anchors[1];
    offset_anchors[2] = shift_x + offset_cell_anchors[2];
    offset_anchors[3] = shift_y + offset_cell_anchors[3];
  }
}

template <typename T>
inline void GetAnchors(
    const int num_anchors,
    const int num_cell_anchors,
    const int num_classes,
    const GridArgs<int64_t>& args,
    const T* cell_anchors,
    const int64_t* indices,
    T* anchors) {
  for (int i = 0; i < num_anchors; ++i) {
    auto index = indices[i];
    index /= num_classes;
    const auto w = index % args.w;
    index /= args.w;
    const auto h = index % args.h;
    index /= args.h;
    const auto shift_x = T(w * args.stride);
    const auto shift_y = T(h * args.stride);
    auto* offset_anchors = anchors + i * 7 + 1;
    const auto* offset_cell_anchors = cell_anchors + index * 4;
    offset_anchors[0] = shift_x + offset_cell_anchors[0];
    offset_anchors[1] = shift_y + offset_cell_anchors[1];
    offset_anchors[2] = shift_x + offset_cell_anchors[2];
    offset_anchors[3] = shift_y + offset_cell_anchors[3];
  }
}

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_ANCHORS_H_
