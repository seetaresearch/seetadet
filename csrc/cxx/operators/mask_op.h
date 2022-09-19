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

#ifndef DRAGON_EXTENSION_OPERATORS_MASK_OP_H_
#define DRAGON_EXTENSION_OPERATORS_MASK_OP_H_

#include <dragon/core/operator.h>

namespace dragon {

template <class Context>
class PasteMaskOp final : public Operator<Context> {
 public:
  PasteMaskOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mask_threshold_(OP_SINGLE_ARG(float, "mask_threshold", 0.5f)) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, sizes);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float mask_threshold_;
  DECLARE_OP_REPEATED_ARG(int64_t, sizes);
};

DEFINE_OP_REPEATED_ARG(int64_t, PasteMaskOp, sizes);

} // namespace dragon

#endif // DRAGON_EXTENSION_OPERATORS_MASK_OP_H_
