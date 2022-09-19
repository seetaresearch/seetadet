#include <dragon/core/context_mps.h>
#include <dragon/core/workspace.h>

#include "../../../utils/detection/nms.h"
#include "../../../utils/detection/utils.h"

namespace dragon {

namespace detection {

namespace {

#define NUM_THREADS 64

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant uint uint_arg1 [[function_constant(0)]];
constant float float_arg1 [[function_constant(1)]];

template <typename T>
bool CheckIoU(const T thresh, device const T* a, threadgroup T* b) {
  const T x1 = max(a[0], b[0]);
  const T y1 = max(a[1], b[1]);
  const T x2 = min(a[2], b[2]);
  const T y2 = min(a[3], b[3]);
  const T width = max(T(0), x2 - x1);
  const T height = max(T(0), y2 - y1);
  const T inter = width * height;
  const T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  const T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return inter >= thresh * (Sa + Sb - inter);
}

template <typename T>
kernel void NonMaxSuppression(
    device const T* boxes,
    device uint64_t* mask,
    const uint2 gridDim [[threadgroups_per_grid]],
    const uint2 blockIdx [[threadgroup_position_in_grid]],
    const uint2 threadIdx [[thread_position_in_threadgroup]]) {
  const uint row_start = blockIdx.y;
  const uint col_start = blockIdx.x;
  if (row_start > col_start) return;
  const uint row_size = min(uint_arg1 - row_start * uint(64), uint(64));
  const uint col_size = min(uint_arg1 - col_start * uint(64), uint(64));
  threadgroup T block_boxes[256];
  if (threadIdx.x < col_size) {
    threadgroup T* offset_block_boxes = block_boxes + threadIdx.x * 4;
    device const T* offset_boxes = boxes + (col_start * uint(64) + threadIdx.x) * 5;
    for (int i = 0; i < 4; ++i) {
      *(offset_block_boxes++) = *(offset_boxes++);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threadIdx.x < row_size) {
    const uint index = row_start * uint(64) + threadIdx.x;
    device const T* offset_boxes = boxes + index * 5;
    uint64_t val = 0;
    const uint start = (row_start == col_start) ? (threadIdx.x + 1) : 0;
    for (uint i = start; i < col_size; ++i) {
      if (CheckIoU(T(float_arg1), offset_boxes, block_boxes + i * 4)) {
        val |= (uint64_t(1) << i);
      }
    }
    mask[index * gridDim.x + col_start] = val;
  }
}

#define INSTANTIATE_KERNEL(T) \
  template [[host_name("NonMaxSuppression_"#T)]] \
  kernel void NonMaxSuppression( \
      device const T*, device uint64_t*, uint2, uint2, uint2);

INSTANTIATE_KERNEL(float);
#undef INSTANTIATE_KERNEL

)";

} // namespace

template <>
void ApplyNMS<float, MPSContext>(
    const int N,
    const int K,
    const int boxes_offset,
    const float thresh,
    const float* boxes,
    vector<int64_t>& indices,
    MPSContext* ctx) {
  const auto num_blocks = utils::DivUp(N, NUM_THREADS);
  auto* NMS_mask = ctx->workspace()->CreateTensor("NMS_mask");
  NMS_mask->Reshape({N * num_blocks});
  auto* mask = reinterpret_cast<uint64_t*>(
      NMS_mask->template mutable_data<int64_t, MPSContext>());
  auto kernel = MPSKernel::TypedString<float>("NonMaxSuppression");
  const uint arg1 = N;
  auto args = vector<MPSConstant>({
      MPSConstant(&arg1, MTLDataTypeUInt, 0),
      MPSConstant(&thresh, MTLDataTypeFloat, 1),
  });
  auto* command_buffer = ctx->mps_stream()->command_buffer();
  auto* encoder = [command_buffer computeCommandEncoder];
  auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args);
  [encoder setComputePipelineState:pso];
  [encoder setBuffer:id<MTLBuffer>(boxes) offset:boxes_offset * 4 atIndex:0];
  [encoder setBuffer:id<MTLBuffer>(mask) offset:0 atIndex:1];
  [encoder dispatchThreadgroups:MTLSizeMake(num_blocks, num_blocks, 1)
          threadsPerThreadgroup:MTLSizeMake(NUM_THREADS, 1, 1)];
  [encoder endEncoding];
  [encoder release];
  ctx->FinishDeviceComputation();
  mask = reinterpret_cast<uint64_t*>(
      const_cast<int64_t*>(NMS_mask->template data<int64_t, CPUContext>()));
  vector<uint64_t> is_dead(num_blocks);
  memset(&is_dead[0], 0, sizeof(uint64_t) * num_blocks);
  int num_selected = 0;
  indices.resize(K);
  for (int i = 0; i < N; ++i) {
    const int nblock = i / NUM_THREADS, inblock = i % NUM_THREADS;
    if (!(is_dead[nblock] & (uint64_t(1) << inblock))) {
      indices[num_selected++] = i;
      if (num_selected >= K) break;
      auto* offset_mask = mask + i * num_blocks;
      for (int j = nblock; j < num_blocks; ++j) {
        is_dead[j] |= offset_mask[j];
      }
    }
  }
  indices.resize(num_selected);
}

} // namespace detection

} // namespace dragon
