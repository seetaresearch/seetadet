#include <dragon/core/context_cuda.h>
#include <dragon/core/workspace.h>

#include "../../../utils/detection/bbox.h"
#include "../../../utils/detection/nms.h"
#include "../../../utils/detection/utils.h"

namespace dragon {

namespace detection {

namespace {

#define NUM_THREADS 64

template <typename T>
__global__ void _NonMaxSuppression(
    const int N,
    const T thresh,
    const T* boxes,
    uint64_t* mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  if (row_start > col_start) return;

  const int row_size = min(N - row_start * NUM_THREADS, NUM_THREADS);
  const int col_size = min(N - col_start * NUM_THREADS, NUM_THREADS);

  __shared__ T block_boxes[NUM_THREADS * 4];

  if (threadIdx.x < col_size) {
    auto* offset_block_boxes = block_boxes + threadIdx.x * 4;
    auto* offset_boxes = boxes + (col_start * NUM_THREADS + threadIdx.x) * 5;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(offset_block_boxes++) = *(offset_boxes++);
    }
  }

  __syncthreads();

  if (threadIdx.x < row_size) {
    const int index = row_start * NUM_THREADS + threadIdx.x;
    const T* offset_boxes = boxes + index * 5;
    uint64_t val = 0;
    const int start = (row_start == col_start) ? (threadIdx.x + 1) : 0;
    for (int i = start; i < col_size; ++i) {
      if (utils::CheckIoU(thresh, offset_boxes, block_boxes + i * 4)) {
        val |= (uint64_t(1) << i);
      }
    }
    mask[index * gridDim.x + col_start] = val;
  }
}

} // namespace

template <>
void ApplyNMS<float, CUDAContext>(
    const int N,
    const int K,
    const int boxes_offset,
    const float thresh,
    const float* boxes,
    vector<int64_t>& indices,
    CUDAContext* ctx) {
  boxes = boxes + boxes_offset;
  const auto num_blocks = utils::DivUp(N, NUM_THREADS);
  auto* NMS_mask = ctx->workspace()->CreateTensor("NMS_mask");
  NMS_mask->Reshape({N * num_blocks});
  auto* mask = reinterpret_cast<uint64_t*>(
      NMS_mask->template mutable_data<int64_t, CUDAContext>());
  vector<uint64_t> mask_host(N * num_blocks);
  _NonMaxSuppression<<<
      dim3(num_blocks, num_blocks),
      NUM_THREADS,
      0,
      ctx->cuda_stream()>>>(N, thresh, boxes, mask);
  CUDA_CHECK(cudaMemcpyAsync(
      mask_host.data(),
      mask,
      mask_host.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost,
      ctx->cuda_stream()));
  ctx->FinishDeviceComputation();
  vector<uint64_t> is_dead(num_blocks);
  memset(&is_dead[0], 0, sizeof(uint64_t) * num_blocks);
  int num_selected = 0;
  indices.resize(K);
  for (int i = 0; i < N; ++i) {
    const int nblock = i / NUM_THREADS, inblock = i % NUM_THREADS;
    if (!(is_dead[nblock] & (uint64_t(1) << inblock))) {
      indices[num_selected++] = i;
      if (num_selected >= K) break;
      auto* offset_mask = &mask_host[0] + i * num_blocks;
      for (int j = nblock; j < num_blocks; ++j) {
        is_dead[j] |= offset_mask[j];
      }
    }
  }
  indices.resize(num_selected);
}

} // namespace detection

} // namespace dragon
