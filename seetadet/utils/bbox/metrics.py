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
"""Bounding-Box metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.utils.bbox import cython_bbox

import numpy as np


def bbox_overlaps(boxes1, boxes2):
    """Return the overlaps between two group of boxes."""
    boxes1 = np.ascontiguousarray(boxes1, dtype=np.float)
    boxes2 = np.ascontiguousarray(boxes2, dtype=np.float)
    return cython_bbox.bbox_overlaps(boxes1, boxes2)


def bbox_centerness(boxes1, boxes2):
    """Return centerness between two group of boxes."""
    ctr_x = (boxes1[:, 2] + boxes1[:, 0]) / 2
    ctr_y = (boxes1[:, 3] + boxes1[:, 1]) / 2
    l = ctr_x - boxes2[:, 0]
    t = ctr_y - boxes2[:, 1]
    r = boxes2[:, 2] - ctr_x
    b = boxes2[:, 3] - ctr_y
    centerness = ((np.minimum(l, r) / np.maximum(l, r)) *
                  (np.minimum(t, b) / np.maximum(t, b)))
    min_dist = np.stack([l, t, r, b], axis=1).min(axis=1)
    keep_inds = np.where(min_dist > 0.01)[0]
    discard_inds = np.where(min_dist <= 0.01)[0]
    centerness[keep_inds] = np.sqrt(centerness[keep_inds])
    centerness[discard_inds] = -1
    return centerness, keep_inds, discard_inds


def boxes_area(boxes):
    """Return the area of boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
