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

#ifndef DRAGON_EXTENSION_OPERATORS_SCALE_OP_H_
#define DRAGON_EXTENSION_OPERATORS_SCALE_OP_H_

#include <dragon/core/operator.h>

namespace dragon {

template <class Context>
class IdentityScaleGradientOp final : public Operator<Context> {
 public:
  IdentityScaleGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OP_SINGLE_ARG(float, "scale", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float scale_;
};

} // namespace dragon

#endif // DRAGON_EXTENSION_OPERATORS_SCALE_OP_H_
