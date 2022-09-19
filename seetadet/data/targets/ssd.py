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
"""Generate targets for SSD head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from seetadet.core.config import cfg
from seetadet.data.build import ANCHOR_SAMPLERS
from seetadet.data.anchors.ssd import AnchorGenerator
from seetadet.data.assigners import MaxIoUAssigner
from seetadet.ops.normalization import to_tensor
from seetadet.utils.bbox import bbox_overlaps
from seetadet.utils.bbox import bbox_transform


@ANCHOR_SAMPLERS.register('ssd')
class AnchorTargets(object):
    """Generate ground-truth targets for anchors."""

    def __init__(self):
        super(AnchorTargets, self).__init__()
        self.generator = AnchorGenerator(
            strides=cfg.ANCHOR_GENERATOR.STRIDES,
            sizes=cfg.ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.ANCHOR_GENERATOR.ASPECT_RATIOS)
        self.assigner = MaxIoUAssigner(
            pos_iou_thr=cfg.SSD.POSITIVE_OVERLAP,
            neg_iou_thr=cfg.SSD.NEGATIVE_OVERLAP,
            gt_max_assign_all=False)
        self.neg_pos_ratio = (1.0 / cfg.SSD.POSITIVE_FRACTION) - 1.0
        max_size = cfg.ANCHOR_GENERATOR.STRIDES[-1]
        self.generator.reset_grid(max_size)

    def sample(self, gt_boxes):
        """Sample positive and negative anchors."""
        anchors = self.generator.grid_anchors
        # Assign ground-truth according to the IoU.
        labels = self.assigner.assign(anchors, gt_boxes)
        # Select positive and non-positive indices.
        return {'fg_inds': np.where(labels > 0)[0],
                'bg_inds': np.where(labels <= 0)[0]}

    def compute(self, **inputs):
        """Compute anchor targets."""
        num_images = len(inputs['gt_boxes'])
        num_anchors = self.generator.grid_anchors.shape[0]
        cls_score = inputs['cls_score'].numpy().astype('float32')
        blobs = collections.defaultdict(list)
        # "1" is positive, "0" is negative, "-1" is don't care
        labels = np.full((num_images, num_anchors,), -1, 'int64')
        for i, gt_boxes in enumerate(inputs['gt_boxes']):
            fg_inds = pos_inds = inputs['fg_inds'][i]
            neg_inds = inputs['bg_inds'][i]
            # Mining hard negatives as background.
            num_pos, num_neg = len(pos_inds), len(neg_inds)
            num_bg = min(int(num_pos * self.neg_pos_ratio), num_neg)
            neg_score = cls_score[i, neg_inds, 0]
            bg_inds = neg_inds[np.argsort(neg_score)][:num_bg]
            # Compute bbox targets.
            anchors = self.generator.grid_anchors[fg_inds]
            gt_assignments = bbox_overlaps(anchors, gt_boxes).argmax(axis=1)
            bbox_targets = bbox_transform(anchors, gt_boxes[gt_assignments, :4],
                                          weights=cfg.SSD.BBOX_REG_WEIGHTS)
            blobs['bbox_anchors'].append(anchors)
            blobs['bbox_targets'].append(bbox_targets)
            # Compute label assignments.
            labels[i, bg_inds] = 0
            labels[i, fg_inds] = gt_boxes[gt_assignments, 4]
            # Compute sparse indices.
            fg_inds += i * num_anchors
            blobs['bbox_inds'].extend([fg_inds])

        return {
            'labels': to_tensor(labels),
            'bbox_inds': to_tensor(np.hstack(blobs['bbox_inds'])),
            'bbox_targets': to_tensor(np.vstack(blobs['bbox_targets'])),
            'bbox_anchors': to_tensor(np.vstack(blobs['bbox_anchors'])),
        }
