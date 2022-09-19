#include <dragon/core/context_mps.h>

#include "../../../utils/detection/mask.h"

namespace dragon {

namespace detection {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant int int_arg1 [[function_constant(0)]];     // H
constant int int_arg2 [[function_constant(1)]];     // W
constant int int_arg3 [[function_constant(2)]];     // mask_h
constant int int_arg4 [[function_constant(3)]];     // mask_w
constant float float_arg1 [[function_constant(4)]]; // thresh

template <typename IndexT>
bool WithinBounds2d(IndexT h, IndexT w, IndexT H, IndexT W) {
  return h >= IndexT(0) && h < H && w >= IndexT(0) && w < W;
}

template <typename T>
kernel void PasteMask(
    device const T* masks,
    device const float* boxes,
    device uint8_t* im,
    const uint index [[thread_position_in_grid]]) {
  const int w = int(index) % int_arg2;
  const int h = int(index) / int_arg2 % int_arg1;
  const int n = int(index) / (int_arg2 * int_arg1);

  device const float* box = boxes + n * 4;
  device const T* mask = masks + n * int_arg3 * int_arg4;

  const float gx = (float(w) + 0.5f - box[0]) / (box[2] - box[0]) * 2.f;
  const float gy = (float(h) + 0.5f - box[1]) / (box[3] - box[1]) * 2.f;
  const float ix = (gx * float(int_arg4) - 1.f) * 0.5f;
  const float iy = (gy * float(int_arg3) - 1.f) * 0.5f;

  const int ix_nw = floor(ix);
  const int iy_nw = floor(iy);
  const int ix_ne = ix_nw + 1;
  const int iy_ne = iy_nw;
  const int ix_sw = ix_nw;
  const int iy_sw = iy_nw + 1;
  const int ix_se = ix_nw + 1;
  const int iy_se = iy_nw + 1;

  T nw = T((ix_se - ix) * (iy_se - iy));
  T ne = T((ix - ix_sw) * (iy_sw - iy));
  T sw = T((ix_ne - ix) * (iy - iy_ne));
  T se = T((ix - ix_nw) * (iy - iy_nw));

  T val = T(0);
  if (WithinBounds2d(iy_nw, ix_nw, int_arg3, int_arg4)) {
    val += mask[iy_nw * int_arg4 + ix_nw] * nw;
  }
  if (WithinBounds2d(iy_ne, ix_ne, int_arg3, int_arg4)) {
    val += mask[iy_ne * int_arg4 + ix_ne] * ne;
  }
  if (WithinBounds2d(iy_sw, ix_sw, int_arg3, int_arg4)) {
    val += mask[iy_sw * int_arg4 + ix_sw] * sw;
  }
  if (WithinBounds2d(iy_se, ix_se, int_arg3, int_arg4)) {
    val += mask[iy_se * int_arg4 + ix_se] * se;
  }
  im[index] = (val >= T(float_arg1) ? uint8_t(1) : uint8_t(0));
}

#define INSTANTIATE_KERNEL(T) \
  template [[host_name("PasteMask_"#T)]] \
  kernel void PasteMask( \
    device const T*, device const float*, device uint8_t*, uint);

INSTANTIATE_KERNEL(float);
#undef INSTANTIATE_KERNEL

)";

} // namespace

template <>
void PasteMask<float, MPSContext>(
    const int N,
    const int H,
    const int W,
    const int mask_h,
    const int mask_w,
    const float thresh,
    const float* masks,
    const float* boxes,
    uint8_t* im,
    MPSContext* ctx) {
  auto kernel = MPSKernel::TypedString<float>("PasteMask");
  auto args = vector<MPSConstant>({
      MPSConstant(&H, MTLDataTypeInt, 0),
      MPSConstant(&W, MTLDataTypeInt, 1),
      MPSConstant(&mask_h, MTLDataTypeInt, 2),
      MPSConstant(&mask_w, MTLDataTypeInt, 3),
      MPSConstant(&thresh, MTLDataTypeFloat, 4),
  });
  auto* command_buffer = ctx->mps_stream()->command_buffer();
  auto* encoder = [command_buffer computeCommandEncoder];
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  [encoder setBuffer:id<MTLBuffer>(masks) offset:0 atIndex:0];
  [encoder setBuffer:id<MTLBuffer>(boxes) offset:0 atIndex:1];
  [encoder setBuffer:id<MTLBuffer>(im) offset:0 atIndex:2];
  MPSDispatchThreads((N * H * W), encoder, pso);
  [encoder endEncoding];
  [encoder release];
}

} // namespace detection

} // namespace dragon
