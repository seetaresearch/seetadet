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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from seetadet.core.config import cfg
from seetadet.data.build import ANCHOR_SAMPLERS
from seetadet.data.anchors.rpn import AnchorGenerator
from seetadet.data.assigners import MaxIoUAssigner
from seetadet.ops.normalization import to_tensor
from seetadet.utils.bbox import bbox_overlaps
from seetadet.utils.bbox import bbox_transform


@ANCHOR_SAMPLERS.register('retinanet')
class AnchorTargets(object):
    """Generate ground-truth targets for anchors."""

    def __init__(self):
        super(AnchorTargets, self).__init__()
        self.generator = AnchorGenerator(
            strides=cfg.ANCHOR_GENERATOR.STRIDES,
            sizes=cfg.ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.ANCHOR_GENERATOR.ASPECT_RATIOS,
            scales_per_octave=3)
        self.assigner = MaxIoUAssigner(
            pos_iou_thr=cfg.RETINANET.POSITIVE_OVERLAP,
            neg_iou_thr=cfg.RETINANET.NEGATIVE_OVERLAP)
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
        # Select foreground and ignored indices
        # to avoid too many backgrounds.
        # (~100x faster for 200k background indices)
        return {'fg_inds': np.where(labels > 0)[0],
                'bg_inds': np.where(labels < 0)[0]}

    def compute(self, **inputs):
        """Compute anchor targets."""
        shapes = [x[:2] for x in inputs['grid_info']]
        num_images = len(inputs['gt_boxes'])
        num_anchors = self.generator.num_anchors(shapes)
        blobs = collections.defaultdict(list)
        # "1" is positive, "0" is negative, "-1" is don't care.
        labels = np.zeros((num_images, num_anchors), 'int64')
        for i, gt_boxes in enumerate(inputs['gt_boxes']):
            fg_inds = inputs['fg_inds'][i]
            ignore_inds = inputs['bg_inds'][i]
            # Narrow anchors to match the feature layout.
            ignore_inds = self.generator.narrow_anchors(shapes, ignore_inds)
            fg_inds, anchors = self.generator.narrow_anchors(shapes, fg_inds, True)
            # Compute bbox targets.
            gt_assignments = bbox_overlaps(anchors, gt_boxes).argmax(axis=1)
            bbox_targets = bbox_transform(anchors, gt_boxes[gt_assignments, :4])
            blobs['bbox_anchors'].append(anchors)
            blobs['bbox_targets'].append(bbox_targets)
            # Compute label assignments.
            labels[i, ignore_inds] = -1
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
