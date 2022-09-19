/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_TYPES_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_TYPES_H_

#include <dragon/core/common.h>

namespace dragon {

namespace detection {

template <typename T>
struct Box4d {
  T x1, y1, x2, y2;
};

template <typename T>
struct Box5d {
  T x1, y1, x2, y2, score;
};

template <typename IndexT>
struct ImageArgs {
  ImageArgs(const float* im_info) {
    h = im_info[0], w = im_info[1];
    scale_h = im_info[2], scale_w = im_info[3];
  }

  IndexT batch_ind, h, w;
  float scale_h, scale_w;
};

template <typename IndexT>
struct GridArgs {
  IndexT h, w, stride, size, offset;
};

template <typename KeyT, typename ValueT>
struct KeyValueMap {
  typedef KeyT key_type;
  typedef ValueT value_type;

  friend void swap(KeyValueMap& x, KeyValueMap& y) {
    std::swap(*x.key_ptr, *y.key_ptr);
    std::swap(*x.value_ptr, *y.value_ptr);
  }

  KeyT* key_ptr = nullptr;
  ValueT* value_ptr = nullptr;
};

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_TYPES_H_
