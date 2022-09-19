#include <dragon/core/context_cuda.h>
#include <dragon/core/workspace.h>
#include <dragon/utils/device/common_thrust.h>

#include "../../../utils/detection/iterator.h"
#include "../../../utils/detection/proposals.h"

namespace dragon {

namespace detection {

namespace {

template <typename KeyT, typename ValueT>
struct ThresholdFunctor {
  ThresholdFunctor(ValueT thresh) : thresh_(thresh) {}
  inline __device__ bool operator()(
      const thrust::tuple<KeyT, ValueT>& kv) const {
    return thrust::get<1>(kv) > thresh_;
  }
  ValueT thresh_;
};

template <typename IterT>
inline void ArgPartition(const int N, const int K, IterT data) {
  std::nth_element(
      data,
      data + K,
      data + N,
      [](const typename IterT::value_type& lhs,
         const typename IterT::value_type& rhs) {
        return *lhs.value_ptr > *rhs.value_ptr;
      });
}

} // namespace

template <>
void SelectTopK<float, CUDAContext>(
    const int N,
    const int K,
    const float thresh,
    const float* scores,
    vector<float>& out_scores,
    vector<int64_t>& out_indices,
    CUDAContext* ctx) {
  int num_selected = N;
  int64_t* indices = nullptr;
  if (thresh > 0.f) {
    indices = ctx->workspace()->data<int64_t, CUDAContext>(N, "BufferKernel");
    auto policy = thrust::cuda::par.on(ctx->cuda_stream());
    auto functor = ThresholdFunctor<int64_t, float>(thresh);
    thrust::sequence(policy, indices, indices + N);
    auto kv = thrust::make_tuple(indices, const_cast<float*>(scores));
    auto first = thrust::make_zip_iterator(kv);
    auto last = thrust::partition(policy, first, first + N, functor);
    num_selected = last - first;
  }
  out_scores.resize(num_selected);
  out_indices.resize(num_selected);
  CUDA_CHECK(cudaMemcpyAsync(
      out_scores.data(),
      scores,
      num_selected * sizeof(float),
      cudaMemcpyDeviceToHost,
      ctx->cuda_stream()));
  if (thresh > 0.f) {
    CUDA_CHECK(cudaMemcpyAsync(
        out_indices.data(),
        indices,
        num_selected * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        ctx->cuda_stream()));
  } else {
    std::iota(out_indices.begin(), out_indices.end(), 0);
  }
  ctx->FinishDeviceComputation();
  if (num_selected > K) {
    auto iter = KeyValueMapIterator<KeyValueMap<int64_t, float>>(
        out_indices.data(), out_scores.data());
    ArgPartition(num_selected, K, iter);
    out_scores.resize(K);
    out_indices.resize(K);
  }
}

} // namespace detection

} // namespace dragon
