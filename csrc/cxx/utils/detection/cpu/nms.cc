#include <dragon/core/context.h>

#include "../../../utils/detection/bbox.h"
#include "../../../utils/detection/nms.h"

namespace dragon {

namespace detection {

template <>
void ApplyNMS<float, CPUContext>(
    const int N,
    const int K,
    const int boxes_offset,
    const float thresh,
    const float* boxes,
    vector<int64_t>& indices,
    CPUContext* ctx) {
  boxes = boxes + boxes_offset;
  int num_selected = 0;
  indices.resize(K);
  vector<char> is_dead(N, 0);
  for (int i = 0; i < N; ++i) {
    if (is_dead[i]) continue;
    indices[num_selected++] = i;
    if (num_selected >= K) break;
    for (int j = i + 1; j < N; ++j) {
      if (is_dead[j]) continue;
      if (!utils::CheckIoU(thresh, &boxes[i * 5], &boxes[j * 5])) continue;
      is_dead[j] = 1;
    }
  }
  indices.resize(num_selected);
}

} // namespace detection

} // namespace dragon
