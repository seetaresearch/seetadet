#include "../operators/retinanet_decoder_op.h"
#include "../utils/detection.h"

namespace dragon {

template <class Context>
template <typename T>
void RetinaNetDecoderOp<Context>::DoRunWithType() {
  auto N = Input(SCORES).dim(0);
  auto AxK = Input(SCORES).dim(1);
  auto C = Input(SCORES).dim(2);
  auto AxKxC = AxK * C;
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

  // Decode detections.
  auto* scores = Input(SCORES).template data<T, Context>();
  auto* deltas = Input(DELTAS).template data<T, CPUContext>();
  auto* im_info = Input(IM_INFO).template data<float, CPUContext>();
  auto* Y = Output(0)->Reshape({N * num_lvls * pre_nms_topk_, 7});
  auto* dets = Y->template mutable_data<float, CPUContext>();

  int64_t size_dets = 0;
  for (int batch_ind = 0; batch_ind < N; ++batch_ind) {
    detection::ImageArgs<int64_t> im_args(im_info + batch_ind * 4);
    im_args.batch_ind = batch_ind;
    for (int lvl_ind = 0; lvl_ind < num_lvls; ++lvl_ind) {
      detection::SelectTopK(
          grid_args[lvl_ind].size * C,
          pre_nms_topk_,
          score_thresh_,
          scores + batch_ind * AxKxC + grid_args[lvl_ind].offset * C,
          scores_,
          indices_,
          ctx());
      auto* offset_dets = dets + size_dets * 7;
      auto num_dets = int64_t(indices_.size());
      size_dets += num_dets;
      detection::GetAnchors(
          num_dets,
          A, // num_cell_anchors
          C, // num_classes
          grid_args[lvl_ind],
          cell_anchors_[lvl_ind].data(),
          indices_.data(),
          offset_dets);
      detection::DecodeDetections(
          num_dets,
          AxK, // num_anchors
          C, // num_classes
          im_args,
          grid_args[lvl_ind],
          scores_.data(),
          deltas + batch_ind * Input(DELTAS).stride(0),
          indices_.data(),
          offset_dets);
    }
  }

  // Shrink to the correct dimensions.
  Y->Reshape({size_dets, 7});
}

DEPLOY_CPU_OPERATOR(RetinaNetDecoder);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RetinaNetDecoder);
#endif
#ifdef USE_MPS
REGISTER_MPS_OPERATOR(RetinaNetDecoder, RetinaNetDecoderOp<CPUContext>);
#endif

OPERATOR_SCHEMA(RetinaNetDecoder).NumInputs(4).NumOutputs(1);

NO_GRADIENT(RetinaNetDecoder);

} // namespace dragon
