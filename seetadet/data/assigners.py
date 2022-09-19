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
"""Ground-truth assigners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from seetadet.utils.bbox import bbox_overlaps


class MaxIoUAssigner(object):
    """Assign ground-truth to boxes according to the IoU."""

    def __init__(
        self,
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.0,
        match_low_quality=True,
        gt_max_assign_all=True,
    ):
        """Create a ``MaxIoUAssigner``.

        Parameters
        ----------
        pos_iou_thr : float, optional, default=0.5
            The minimum IoU overlap to label positives.
        neg_iou_thr : float, optional, default=0.5
            The maximum IoU overlap to label negatives.
        min_pos_iou : float, optional, default=0.0
            The minimum IoU overlap to match low quality.
        match_low_quality : bool, optional, default=True
            Assign boxes for each gt box or not.
        gt_max_assign_all : bool, optional, default=True
            Assign all boxes with max overlaps for gt boxes or not.

        """
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.gt_max_assign_all = gt_max_assign_all

    def assign(self, boxes, gt_boxes):
        # Initialize assigns with ignored index "-1".
        num_boxes = len(boxes)
        labels = np.empty((num_boxes,), 'int8')
        labels.fill(-1)

        # Overlaps between the anchors and the gt boxes.
        overlaps = bbox_overlaps(boxes, gt_boxes)
        max_overlaps = overlaps.max(axis=1)

        # Background: below threshold IoU.
        labels[max_overlaps < self.neg_iou_thr] = 0

        # Foreground: above threshold IoU.
        labels[max_overlaps >= self.pos_iou_thr] = 1

        # Foreground: for each gt, assign anchor(s) with highest overlap.
        if self.match_low_quality:
            if self.gt_max_assign_all:
                gt_max_overlaps = overlaps.max(axis=0)
                if self.min_pos_iou > 0:
                    for i in np.where(gt_max_overlaps >= self.min_pos_iou)[0]:
                        labels[overlaps[:, i] == gt_max_overlaps[i]] = 1
                else:
                    labels[np.where(overlaps == gt_max_overlaps)[0]] = 1
            else:
                labels[overlaps.argmax(axis=0)] = 1

        # Return the assigned labels for future development.
        return labels
