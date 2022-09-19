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
"""Helper functions for bounding box."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def clip_boxes(boxes, im_shape):
    """Clip the boxes."""
    xmax, ymax = im_shape[1], im_shape[0]
    boxes[:, (0, 2)] = np.maximum(np.minimum(boxes[:, (0, 2)], xmax), 0)
    boxes[:, (1, 3)] = np.maximum(np.minimum(boxes[:, (1, 3)], ymax), 0)
    return boxes


def clip_tiled_boxes(boxes, im_shape):
    """Clip the tiled boxes."""
    xmax, ymax = im_shape[1], im_shape[0]
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], xmax), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], ymax), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], xmax), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], ymax), 0)
    return boxes


def flip_boxes(boxes, width):
    """Flip the boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0] = width - boxes[:, 2]
    boxes_flipped[:, 2] = width - boxes[:, 0]
    return boxes_flipped


def filter_empty_boxes(boxes):
    """Return the indices of non-empty boxes."""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return np.where((ws > 0) & (hs > 0))[0]


def distribute_boxes(boxes, lvl_min, lvl_max):
    """Return the fpn level of boxes."""
    if len(boxes) == 0:
        return []
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    s = np.sqrt(ws * hs)
    s0 = 224  # default: 224
    lvl0 = 4  # default: 4
    lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    return np.clip(lvls, lvl_min, lvl_max)
