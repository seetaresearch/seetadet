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
"""Generate targets for RPN head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import numpy.random as npr

from seetadet.core.config import cfg
from seetadet.data.anchors.rpn import AnchorGenerator
from seetadet.data.assigners import MaxIoUAssigner
from seetadet.data.build import ANCHOR_SAMPLERS
from seetadet.ops.normalization import to_tensor
from seetadet.utils.bbox import bbox_overlaps
from seetadet.utils.bbox import bbox_transform


@ANCHOR_SAMPLERS.register(['faster_rcnn', 'mask_rcnn', 'cascade_rcnn'])
class AnchorTargets(object):
    """Generate ground-truth targets for anchors."""

    def __init__(self):
        super(AnchorTargets, self).__init__()
        self.generator = AnchorGenerator(
            strides=cfg.ANCHOR_GENERATOR.STRIDES,
            sizes=cfg.ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.ANCHOR_GENERATOR.ASPECT_RATIOS)
        self.assigner = MaxIoUAssigner(
            pos_iou_thr=cfg.RPN.POSITIVE_OVERLAP,
            neg_iou_thr=cfg.RPN.NEGATIVE_OVERLAP)
        max_size = max(cfg.TRAIN.MAX_SIZE, max(cfg.TRAIN.SCALES))
        if cfg.BACKBONE.COARSEST_STRIDE > 0:
            stride = float(cfg.BACKBONE.COARSEST_STRIDE)
            max_size = int(np.ceil(max_size / stride) * stride)
        self.generator.reset_grid(max_size)

    def sample(self, gt_boxes):
        """Sample positive and negative anchors."""
        anchors = self.generator.grid_anchors
        # Assign ground-truth according to the IoU.
        labels = self.assigner.assign(anchors, gt_boxes)
        fg_inds = np.where(labels > 0)[0]
        bg_inds = np.where(labels == 0)[0]
        # Sample sufficient negative labels.
        num_bg = cfg.RPN.BATCH_SIZE * 8
        if len(bg_inds) > num_bg:
            bg_inds = npr.choice(bg_inds, num_bg, False)
        # Select foreground and background indices.
        return {'fg_inds': fg_inds, 'bg_inds': bg_inds}

    def compute(self, **inputs):
        """Compute anchor targets."""
        shapes = [x[:2] for x in inputs['grid_info']]
        num_anchors = self.generator.num_anchors(shapes)
        blobs = collections.defaultdict(list)
        for i, gt_boxes in enumerate(inputs['gt_boxes']):
            fg_inds = inputs['fg_inds'][i]
            bg_inds = inputs['bg_inds'][i]
            # Narrow anchors to match the feature layout.
            bg_inds = self.generator.narrow_anchors(shapes, bg_inds)
            fg_inds, anchors = self.generator.narrow_anchors(shapes, fg_inds, True)
            num_fg = int(cfg.RPN.POSITIVE_FRACTION * cfg.RPN.BATCH_SIZE)
            if len(fg_inds) > num_fg:
                keep = npr.choice(np.arange(len(fg_inds)), num_fg, False)
                fg_inds, anchors = fg_inds[keep], anchors[keep]
            # Sample negative labels if we have too many.
            num_bg = cfg.RPN.BATCH_SIZE - len(fg_inds)
            if len(bg_inds) > num_bg:
                bg_inds = npr.choice(bg_inds, num_bg, False)
            # Compute bbox targets.
            gt_assignments = bbox_overlaps(anchors, gt_boxes).argmax(axis=1)
            bbox_targets = bbox_transform(anchors, gt_boxes[gt_assignments, :4])
            blobs['bbox_anchors'].append(anchors)
            blobs['bbox_targets'].append(bbox_targets)
            # Compute sparse indices.
            fg_inds += i * num_anchors
            bg_inds += i * num_anchors
            blobs['cls_inds'] += [fg_inds, bg_inds]
            blobs['bbox_inds'] += [fg_inds]
            blobs['labels'] += [np.ones_like(fg_inds, 'float32'),
                                np.zeros_like(bg_inds, 'float32')]

        return {
            'labels': to_tensor(np.hstack(blobs['labels'])),
            'cls_inds': to_tensor(np.hstack(blobs['cls_inds'])),
            'bbox_inds': to_tensor(np.hstack(blobs['bbox_inds'])),
            'bbox_targets': to_tensor(np.vstack(blobs['bbox_targets'])),
            'bbox_anchors': to_tensor(np.vstack(blobs['bbox_anchors'])),
        }
