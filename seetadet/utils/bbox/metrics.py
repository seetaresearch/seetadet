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


def bbox_ctrness(boxes1, boxes2):
    """Return centerness between two group of boxes."""
    ctr_x = (boxes1[:, 2] + boxes1[:, 0]) / 2
    ctr_y = (boxes1[:, 3] + boxes1[:, 1]) / 2
    l, t = ctr_x - boxes2[:, 0], ctr_y - boxes2[:, 1]
    r, b = boxes2[:, 2] - ctr_x, boxes2[:, 3] - ctr_y
    ctrness = ((np.minimum(l, r) / np.maximum(l, r)) *
               (np.minimum(t, b) / np.maximum(t, b)))
    return np.sqrt(ctrness)


def boxes_area(boxes):
    """Return the area of boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def boxes_center(boxes):
    """Return the center of boxes."""
    ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
    ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2
    return np.stack([ctr_x, ctr_y], axis=1)


def boxes_point_dist(boxes, points):
    """Return the distance between point and box corners."""
    x1, y1, x2, y2 = np.split(boxes[:, :4], 4, axis=1)
    x, y = np.split(np.expand_dims(points, 1), 2, axis=2)
    return np.concatenate([x - x1, y - y1, x2 - x, y2 - y], axis=2)
