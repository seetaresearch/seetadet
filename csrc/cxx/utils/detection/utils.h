/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_EXTENSION_UTILS_DETECTION_UTILS_H_
#define DRAGON_EXTENSION_UTILS_DETECTION_UTILS_H_

namespace dragon {

namespace detection {

/*
 * Detection Utilities.
 */

namespace utils {

template <typename T>
inline T DivUp(const T a, const T b) {
  return (a + b - T(1)) / b;
}

} // namespace utils

} // namespace detection

} // namespace dragon

#endif // DRAGON_EXTENSION_UTILS_DETECTION_UTILS_H_
