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

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_NMS_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_NMS_H_

#include "../../utils/detection/types.h"

namespace dragon {

namespace detection {

template <typename T, class Context>
void ApplyNMS(
    const int N,
    const int K,
    const int boxes_offset,
    const T thresh,
    const T* boxes,
    vector<int64_t>& indices,
    Context* ctx);

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_NMS_H_
