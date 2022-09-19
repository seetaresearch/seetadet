#include "../operators/nms_op.h"
#include "../utils/detection.h"

namespace dragon {

template <class Context>
template <typename T>
void NonMaxSuppressionOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CHECK(X.ndim() == 2 && X.dim(1) == 5)
      << "\nThe dimensions of boxes should be (num_boxes, 5).";
  detection::ApplyNMS(
      X.dim(0),
      X.dim(0),
      0,
      iou_threshold_,
      X.template mutable_data<T, Context>(),
      out_indices_,
      ctx());
  Y->template CopyFrom<int64_t>(out_indices_);
}

DEPLOY_CPU_OPERATOR(NonMaxSuppression);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(NonMaxSuppression);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(NonMaxSuppression, NonMaxSuppression);
#endif

OPERATOR_SCHEMA(NonMaxSuppression).NumInputs(1).NumOutputs(1);

NO_GRADIENT(NonMaxSuppression);

} // namespace dragon
