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

#ifndef DRAGON_EXTENSION_OPERATORS_NMS_OP_H_
#define DRAGON_EXTENSION_OPERATORS_NMS_OP_H_

#include <dragon/core/operator.h>

namespace dragon {

template <class Context>
class NonMaxSuppressionOp final : public Operator<Context> {
 public:
  NonMaxSuppressionOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        iou_threshold_(OP_SINGLE_ARG(float, "iou_threshold", 0.5f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float iou_threshold_;
  vector<int64_t> out_indices_;
};

} // namespace dragon

#endif // DRAGON_EXTENSION_OPERATORS_NMS_OP_H_
