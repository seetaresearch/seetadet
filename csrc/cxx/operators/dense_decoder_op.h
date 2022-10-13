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

#ifndef DRAGON_EXTENSION_OPERATORS_DENSE_DECODER_OP_H_
#define DRAGON_EXTENSION_OPERATORS_DENSE_DECODER_OP_H_

#include <dragon/core/operator.h>

namespace dragon {

template <class Context>
class DenseDecoderOp final : public Operator<Context> {
 public:
  DenseDecoderOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        strides_(OP_REPEATED_ARG(int64_t, "strides")),
        ratios_(OP_REPEATED_ARG(float, "ratios")),
        scales_(OP_REPEATED_ARG(float, "scales")),
        pre_nms_topk_(OP_SINGLE_ARG(int64_t, "pre_nms_topk", 1000)),
        score_thresh_(OP_SINGLE_ARG(float, "score_thresh", 0.05f)),
        transform_type_(OP_SINGLE_ARG(string, "transform_type", "default")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float>>::Call(this, Input(SCORES));
  }

  template <typename T>
  void DoRunWithType();

  enum INPUT_TAGS { SCORES = 0, DELTAS = 1, IM_INFO = 2, GRID_INFO = 3 };

 protected:
  float score_thresh_;
  vector<int64_t> strides_;
  vector<float> ratios_, scales_;
  int64_t pre_nms_topk_;
  string transform_type_;

  vector<float> scores_;
  vector<int64_t> indices_;
  vector<vector<float>> cell_anchors_;
};

} // namespace dragon

#endif // DRAGON_EXTENSION_OPERATORS_DENSE_DECODER_OP_H_
