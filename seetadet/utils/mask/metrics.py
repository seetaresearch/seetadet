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
"""Mask metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def mask_overlap(box1, box2, mask1, mask2):
    """Compute the overlap of two masks."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    w = x2 - x1
    h = y2 - y1
    # Get masks in the intersection part.
    start_ya = y1 - box1[1]
    start_xa = x1 - box1[0]
    inter_mask_a = mask1[start_ya: start_ya + h, start_xa:start_xa + w]
    start_yb = y1 - box2[1]
    start_xb = x1 - box2[0]
    inter_mask_b = mask2[start_yb: start_yb + h, start_xb:start_xb + w]
    assert inter_mask_a.shape == inter_mask_b.shape
    inter = np.logical_and(inter_mask_b, inter_mask_a).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.:
        return 0.
    return float(inter) / float(union)
