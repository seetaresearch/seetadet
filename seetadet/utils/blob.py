# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Blob utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def blob_vstack(arrays, fill_value=None, dtype=None, size=None, align=None):
    """Stack arrays in sequence vertically."""
    if fill_value is None:
        return np.vstack(arrays)

    # Compute the max stack shape.
    max_shape = np.max(np.stack([arr.shape for arr in arrays]), 0)
    if size is not None and min(size) > 0:
        max_shape[:len(size)] = size
    if align is not None and min(align) > 0:
        align_size = np.ceil(max_shape[:len(align)] / align)
        max_shape[:len(align)] = align_size.astype('int64') * align

    # Fill output with the given value.
    output_dtype = dtype or arrays[0].dtype
    output_shape = [len(arrays)] + list(max_shape)
    output = np.empty(output_shape, output_dtype)
    output[:] = fill_value

    # Copy arrays.
    for i, arr in enumerate(arrays):
        copy_slices = (slice(0, d) for d in arr.shape)
        output[(i,) + tuple(copy_slices)] = arr

    return output
