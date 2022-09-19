#include <dragon/core/workspace.h>

#include "../operators/mask_op.h"
#include "../utils/detection.h"

namespace dragon {

template <class Context>
template <typename T>
void PasteMaskOp<Context>::DoRunWithType() {
  auto &X_masks = Input(0), &X_boxes = Input(1), *Y = Output(0);

  vector<int64_t> Y_dims({X_masks.dim(0)});
  int num_sizes;
  sizes(0, &num_sizes);
  for (int i = 0; i < num_sizes; ++i) {
    Y_dims.push_back(sizes(i));
  }

  if (num_sizes == 2) {
    detection::PasteMask(
        Y_dims[0], // N
        Y_dims[1], // H
        Y_dims[2], // W
        X_masks.dim(1), // mask_h
        X_masks.dim(2), // mask_w
        mask_threshold_,
        X_masks.template data<T, Context>(),
        X_boxes.template data<float, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<uint8_t, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "PasteMask" << num_sizes << "d is not supported.";
  }
}

DEPLOY_CPU_OPERATOR(PasteMask);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PasteMask);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(PasteMask, PasteMask);
#endif

OPERATOR_SCHEMA(PasteMask).NumInputs(2).NumOutputs(1);

NO_GRADIENT(PasteMask);

} // namespace dragon
