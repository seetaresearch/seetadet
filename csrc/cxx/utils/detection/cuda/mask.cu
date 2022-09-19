#include <dragon/core/context_cuda.h>

#include "../../../utils/detection/mask.h"

namespace dragon {

namespace detection {

namespace {

template <typename IndexT>
inline __device__ bool WithinBounds2d(IndexT h, IndexT w, IndexT H, IndexT W) {
  return h >= IndexT(0) && h < H && w >= IndexT(0) && w < W;
}

template <typename T>
__global__ void _PasteMask(
    const int nthreads,
    const int H,
    const int W,
    const int mask_h,
    const int mask_w,
    const T thresh,
    const T* masks,
    const float* boxes,
    uint8_t* im) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % W;
    const int h = index / W % H;
    const int n = index / (H * W);

    const float* box = boxes + n * 4;
    const T* mask = masks + n * mask_h * mask_w;

    const float gx = (float(w) + 0.5f - box[0]) / (box[2] - box[0]) * 2.f;
    const float gy = (float(h) + 0.5f - box[1]) / (box[3] - box[1]) * 2.f;
    const float ix = (gx * float(mask_w) - 1.f) * 0.5f;
    const float iy = (gy * float(mask_h) - 1.f) * 0.5f;

    const int ix_nw = floorf(ix);
    const int iy_nw = floorf(iy);
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
    if (WithinBounds2d(iy_nw, ix_nw, mask_h, mask_w)) {
      val += mask[iy_nw * mask_w + ix_nw] * nw;
    }
    if (WithinBounds2d(iy_ne, ix_ne, mask_h, mask_w)) {
      val += mask[iy_ne * mask_w + ix_ne] * ne;
    }
    if (WithinBounds2d(iy_sw, ix_sw, mask_h, mask_w)) {
      val += mask[iy_sw * mask_w + ix_sw] * sw;
    }
    if (WithinBounds2d(iy_se, ix_se, mask_h, mask_w)) {
      val += mask[iy_se * mask_w + ix_se] * se;
    }
    im[index] = (val >= thresh ? uint8_t(1) : uint8_t(0));
  }
}

} // namespace

template <>
void PasteMask<float, CUDAContext>(
    const int N,
    const int H,
    const int W,
    const int mask_h,
    const int mask_w,
    const float thresh,
    const float* masks,
    const float* boxes,
    uint8_t* im,
    CUDAContext* ctx) {
  const auto NxHxW = N * H * W;
  _PasteMask<<<CUDA_BLOCKS(NxHxW), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      NxHxW, H, W, mask_h, mask_w, thresh, masks, boxes, im);
}

} // namespace detection

} // namespace dragon
