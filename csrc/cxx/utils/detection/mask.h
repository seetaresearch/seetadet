

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

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_MASK_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_MASK_H_

#include "../../utils/detection/types.h"

namespace dragon {

namespace detection {

/*
 * Mask Functions.
 */

template <typename T, class Context>
void PasteMask(
    const int N,
    const int H,
    const int W,
    const int mask_h,
    const int mask_w,
    const float thresh,
    const T* masks,
    const float* boxes,
    uint8_t* im,
    Context* ctx);

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_MASK_H_
