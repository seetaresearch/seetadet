#include <dragon/core/workspace.h>
#include <dragon/operators/array/reshape_op.h>
#include <dragon/utils/math_functions.h>

#include "../operators/scale_op.h"

namespace dragon {

template <class Context>
template <typename T>
void IdentityScaleGradientOp<Context>::DoRunWithType() {
  for (int i = 0; i < InputSize(); ++i) {
    auto &dY = Input(i), *dX = Output(i);
    math::Scale(
        dY.count(),
        scale_,
        dY.template data<T, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  }
}

REGISTER_CPU_OPERATOR(IdentityScale, IdentityOp<CPUContext>);
DEPLOY_CPU_OPERATOR(IdentityScaleGradient);
#ifdef USE_CUDA
REGISTER_CUDA_OPERATOR(IdentityScale, IdentityOp<CUDAContext>);
DEPLOY_CUDA_OPERATOR(IdentityScaleGradient);
#endif
#ifdef USE_MPS
REGISTER_MPS_OPERATOR(IdentityScale, IdentityOp<MPSContext>);
DEPLOY_MPS_OPERATOR(IdentityScaleGradient, IdentityScaleGradient);
#endif

OPERATOR_SCHEMA(IdentityScale).AllowInplace([](int, int) -> bool {
  return true;
});
OPERATOR_SCHEMA(IdentityScaleGradient).AllowInplace([](int, int) -> bool {
  return true;
});
REGISTER_GRADIENT(IdentityScale, SimpleGradientMaker);

} // namespace dragon
