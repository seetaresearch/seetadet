#include "../operators/rpn_decoder_op.h"
#include "../utils/detection.h"

namespace dragon {

template <class Context>
template <typename T>
void RPNDecoderOp<Context>::DoRunWithType() {
  auto N = Input(SCORES).dim(0);
  auto AxK = Input(SCORES).dim(1);
  auto A = int64_t(ratios_.size() * scales_.size());
  auto num_lvls = int64_t(strides_.size());

  // Generate anchors.
  CHECK_EQ(Input(GRID_INFO).dim(0), num_lvls);
  cell_anchors_.resize(strides_.size());
  vector<detection::GridArgs<int64_t>> grid_args(strides_.size());
  for (int i = 0; i < strides_.size(); ++i) {
    grid_args[i].stride = strides_[i];
    auto& anchors = cell_anchors_[i];
    if (int64_t(anchors.size()) == A * 4) continue;
    anchors.resize(A * 4);
    detection::GenerateAnchors(
        strides_[i],
        int64_t(ratios_.size()),
        int64_t(scales_.size()),
        ratios_.data(),
        scales_.data(),
        anchors.data());
  }

  // Set grid arguments.
  auto* grid_info = Input(GRID_INFO).template data<int64_t, CPUContext>();
  detection::SetGridArgs(AxK, A, grid_info, grid_args);

  // Decode proposals.
  auto* scores = Input(SCORES).template data<T, CPUContext>();
  auto* deltas = Input(DELTAS).template data<T, CPUContext>();
  auto* im_info = Input(IM_INFO).template data<float, CPUContext>();
  auto* Y = Output("Y")->Reshape({N * num_lvls * pre_nms_topk_, 5});
  auto* dets = Y->template mutable_data<float, CPUContext>();

  for (int batch_ind = 0; batch_ind < N; ++batch_ind) {
    detection::ImageArgs<int64_t> im_args(im_info + batch_ind * 4);
    im_args.batch_ind = batch_ind;
    for (int lvl_ind = 0; lvl_ind < num_lvls; ++lvl_ind) {
      detection::SelectTopK(
          grid_args[lvl_ind].size,
          pre_nms_topk_,
          0.f,
          scores + batch_ind * AxK + grid_args[lvl_ind].offset,
          scores_,
          indices_,
          (CPUContext*)nullptr); // Faster.
      indices_.resize(pre_nms_topk_, indices_.back());
      auto* offset_dets = dets + lvl_ind * pre_nms_topk_ * 5;
      detection::GetAnchors(
          pre_nms_topk_,
          A, // num_cell_anchors
          grid_args[lvl_ind],
          cell_anchors_[lvl_ind].data(),
          indices_.data(),
          offset_dets);
      detection::DecodeProposals(
          pre_nms_topk_,
          AxK, // num_anchors
          im_args,
          grid_args[lvl_ind],
          scores_.data(),
          deltas + batch_ind * Input(DELTAS).stride(0),
          indices_.data(),
          offset_dets);
      detection::SortBoxes<T, detection::Box5d<T>>(pre_nms_topk_, offset_dets);
    }
  }

  // Apply NMS.
  auto* dets_v2 = Y->template data<float, Context>();
  int64_t size_rois = 0;
  scores_.resize(N * post_nms_topk_);
  indices_.resize(N * post_nms_topk_);
  for (int batch_ind = 0; batch_ind < N; ++batch_ind) {
    std::priority_queue<std::pair<float, int64_t>> pq;
    for (int lvl_ind = 0; lvl_ind < num_lvls; ++lvl_ind) {
      const auto offset = lvl_ind * pre_nms_topk_;
      detection::ApplyNMS(
          pre_nms_topk_, // N
          pre_nms_topk_, // K
          offset * 5, // boxes_offset
          nms_thresh_,
          dets_v2,
          nms_indices_,
          ctx());
      for (size_t i = 0; i < nms_indices_.size(); ++i) {
        const auto index = nms_indices_[i] + offset;
        pq.push(std::make_pair(*(dets + index * 5 + 4), index));
      }
    }
    for (int i = 0; i < post_nms_topk_ && !pq.empty(); ++i) {
      scores_[size_rois] = batch_ind;
      indices_[size_rois++] = pq.top().second;
      pq.pop();
    }
  }

  // Apply Histogram.
  detection::ApplyHistogram(
      size_rois,
      min_level_,
      max_level_,
      canonical_level_,
      canonical_scale_,
      dets,
      scores_.data(),
      indices_.data(),
      output_rois_);

  // Copy to outputs.
  for (int i = 0; i < OutputSize(); ++i) {
    const auto& rois = output_rois_[i];
    vector<int64_t> dims({int64_t(rois.size()) / 5, 5});
    auto* Yi = Output(i)->Reshape(dims);
    std::memcpy(
        Yi->template mutable_data<T, CPUContext>(),
        rois.data(),
        sizeof(T) * rois.size());
  }
}

DEPLOY_CPU_OPERATOR(RPNDecoder);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RPNDecoder);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(RPNDecoder, RPNDecoder);
#endif

OPERATOR_SCHEMA(RPNDecoder).NumInputs(4).NumOutputs(1, INT_MAX);

NO_GRADIENT(RPNDecoder);

} // namespace dragon
