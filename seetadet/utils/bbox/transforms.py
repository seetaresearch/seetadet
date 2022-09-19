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
"""Bounding-Box transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

_DEFAULT_SCALE_CLIP = np.log(1000.0 / 16.0)


def bbox_transform(src_boxes, tgt_boxes, weights=(1., 1., 1., 1.)):
    """Return the bbox transformation deltas."""
    src_widths = src_boxes[:, 2] - src_boxes[:, 0]
    src_heights = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights
    tgt_widths = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_heights = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_widths
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_heights
    (wx, wy, ww, wh), deltas = weights, []
    deltas += [wx * (tgt_ctr_x - src_ctr_x) / src_widths]
    deltas += [wy * (tgt_ctr_y - src_ctr_y) / src_heights]
    deltas += [ww * np.log(tgt_widths / src_widths)]
    deltas += [wh * np.log(tgt_heights / src_heights)]
    return np.vstack(deltas).transpose()


def bbox_transform_inv(boxes, deltas, weights=(1., 1., 1., 1.)):
    """Return the boxes transformed from deltas."""
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    dw = np.minimum(dw, _DEFAULT_SCALE_CLIP)
    dh = np.minimum(dh, _DEFAULT_SCALE_CLIP)
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros(deltas.shape, deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes
