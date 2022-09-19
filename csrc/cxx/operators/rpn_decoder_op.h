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

#ifndef DRAGON_EXTENSION_OPERATORS_RPN_DECODER_OP_H_
#define DRAGON_EXTENSION_OPERATORS_RPN_DECODER_OP_H_

#include <dragon/core/operator.h>

namespace dragon {

template <class Context>
class RPNDecoderOp final : public Operator<Context> {
 public:
  RPNDecoderOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        strides_(OP_REPEATED_ARG(int64_t, "strides")),
        ratios_(OP_REPEATED_ARG(float, "ratios")),
        scales_(OP_REPEATED_ARG(float, "scales")),
        pre_nms_topk_(OP_SINGLE_ARG(int64_t, "pre_nms_topk", 1000)),
        post_nms_topk_(OP_SINGLE_ARG(int64_t, "post_nms_topk", 1000)),
        nms_thresh_(OP_SINGLE_ARG(float, "nms_thresh", 0.7f)),
        min_level_(OP_SINGLE_ARG(int64_t, "min_level", 2)),
        max_level_(OP_SINGLE_ARG(int64_t, "max_level", 5)),
        canonical_level_(OP_SINGLE_ARG(int64_t, "canonical_level", 4)),
        canonical_scale_(OP_SINGLE_ARG(int64_t, "canonical_scale", 224)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float>>::Call(this, Input(SCORES));
  }

  template <typename T>
  void DoRunWithType();

  enum INPUT_TAGS { SCORES = 0, DELTAS = 1, IM_INFO = 2, GRID_INFO = 3 };

 protected:
  float nms_thresh_;
  vector<int64_t> strides_;
  vector<float> ratios_, scales_;
  int64_t min_level_, max_level_;
  int64_t pre_nms_topk_, post_nms_topk_;
  int64_t canonical_level_, canonical_scale_;

  vector<float> scores_;
  vector<int64_t> indices_, nms_indices_;
  vector<vector<float>> cell_anchors_;
  vector<vector<float>> output_rois_;
};

} // namespace dragon

#endif // DRAGON_EXTENSION_OPERATORS_RPN_DECODER_OP_H_
