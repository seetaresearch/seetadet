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

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_PROPOSALS_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_PROPOSALS_H_

#include "../../utils/detection/bbox.h"
#include "../../utils/detection/types.h"

namespace dragon {

namespace detection {

template <typename T, class Context>
void SelectTopK(
    const int N,
    const int K,
    const float thresh,
    const T* input_scores,
    vector<T>& output_scores,
    vector<int64_t>& output_indices,
    Context* ctx);

template <typename T>
void DecodeProposals(
    const int num_proposals,
    const int num_anchors,
    const ImageArgs<int64_t>& im_args,
    const GridArgs<int64_t>& grid_args,
    const T* scores,
    const T* deltas,
    const int64_t* indices,
    T* proposals) {
  T* offset_proposals = proposals;
  const int64_t index_min = grid_args.offset;
  const T* offset_dx = deltas;
  const T* offset_dy = deltas + num_anchors;
  const T* offset_dw = deltas + num_anchors * 2;
  const T* offset_dh = deltas + num_anchors * 3;
  for (int i = 0; i < num_proposals; ++i) {
    const auto index = indices[i] + index_min;
    utils::BBoxTransform(
        offset_dx[index],
        offset_dy[index],
        offset_dw[index],
        offset_dh[index],
        T(im_args.w),
        T(im_args.h),
        T(1),
        T(1),
        offset_proposals);
    offset_proposals[4] = scores[i];
    offset_proposals += 5;
  }
}

template <typename T>
void DecodeDetections(
    const int num_dets,
    const int num_anchors,
    const int num_classes,
    const ImageArgs<int64_t>& im_args,
    const GridArgs<int64_t>& grid_args,
    const T* scores,
    const T* deltas,
    const int64_t* indices,
    T* dets) {
  T* offset_dets = dets;
  const int64_t index_min = num_classes * grid_args.offset;
  const T* offset_dx = deltas;
  const T* offset_dy = deltas + num_anchors;
  const T* offset_dw = deltas + num_anchors * 2;
  const T* offset_dh = deltas + num_anchors * 3;
  for (int i = 0; i < num_dets; ++i) {
    const auto index = (indices[i] + index_min) / num_classes;
    utils::BBoxTransform(
        offset_dx[index],
        offset_dy[index],
        offset_dw[index],
        offset_dh[index],
        T(im_args.w),
        T(im_args.h),
        T(im_args.scale_h),
        T(im_args.scale_w),
        offset_dets + 1);
    offset_dets[0] = T(im_args.batch_ind);
    offset_dets[5] = scores[i];
    offset_dets[6] = T((indices[i] + index_min) % num_classes + 1);
    offset_dets += 7;
  }
}

template <typename T>
inline void ApplyHistogram(
    const int N,
    const int lvl_min,
    const int lvl_max,
    const int lvl0,
    const int s0,
    const T* boxes,
    const T* batch_indices,
    const int64_t* box_indices,
    vector<vector<T>>& output_rois) {
  int K = 0;
  vector<int> keep_indices(N), bin_indices(N);
  vector<int> bin_count(lvl_max - lvl_min + 1, 0);
  for (int i = 0; i < N; ++i) {
    const T* offset_boxes = boxes + box_indices[i] * 5;
    auto lvl = utils::GetBBoxLevel(lvl_min, lvl_max, lvl0, s0, offset_boxes);
    if (lvl < 0) continue; // Empty.
    keep_indices[K++] = i;
    bin_indices[i] = lvl - lvl_min;
    bin_count[lvl - lvl_min]++;
  }
  keep_indices.resize(K);
  output_rois.resize(lvl_max - lvl_min + 1);
  for (int i = 0; i < output_rois.size(); ++i) {
    auto& rois = output_rois[i];
    rois.resize(std::max(bin_count[i], 1) * 5, T(0));
    if (bin_count[i] == 0) rois[0] = T(-1); // Ignored.
  }
  for (auto i : keep_indices) {
    const T* offset_boxes = boxes + box_indices[i] * 5;
    const auto bin_index = bin_indices[i];
    const auto roi_index = --bin_count[bin_index];
    auto& rois = output_rois[bin_index];
    T* offset_rois = rois.data() + roi_index * 5;
    offset_rois[0] = batch_indices[i];
    offset_rois[1] = offset_boxes[0];
    offset_rois[2] = offset_boxes[1];
    offset_rois[3] = offset_boxes[2];
    offset_rois[4] = offset_boxes[3];
  }
}

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_PROPOSALS_H_
