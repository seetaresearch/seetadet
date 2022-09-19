#include <dragon/core/context.h>

#include "../../../utils/detection/proposals.h"

namespace dragon {

namespace detection {

namespace {

template <typename KeyT, typename ValueT>
inline void
ArgPartition(const int N, const int K, const ValueT* values, KeyT* keys) {
  std::nth_element(keys, keys + K, keys + N, [&values](KeyT lhs, KeyT rhs) {
    return values[lhs] > values[rhs];
  });
}

} // namespace

template <>
void SelectTopK<float, CPUContext>(
    const int N,
    const int K,
    const float thresh,
    const float* scores,
    vector<float>& out_scores,
    vector<int64_t>& out_indices,
    CPUContext* ctx) {
  int num_selected = 0;
  out_indices.resize(N);
  if (thresh > 0.f) {
    for (int i = 0; i < N; ++i) {
      if (scores[i] > thresh) {
        out_indices[num_selected++] = i;
      }
    }
  } else {
    num_selected = N;
    std::iota(out_indices.begin(), out_indices.end(), 0);
  }
  if (num_selected > K) {
    ArgPartition(num_selected, K, scores, out_indices.data());
    out_scores.resize(K);
    out_indices.resize(K);
    for (int i = 0; i < K; ++i) {
      out_scores[i] = scores[out_indices[i]];
    }
  } else {
    out_scores.resize(num_selected);
    out_indices.resize(num_selected);
    for (int i = 0; i < num_selected; ++i) {
      out_scores[i] = scores[out_indices[i]];
    }
  }
}

} // namespace detection

} // namespace dragon
