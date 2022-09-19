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
import numpy.random as npr

from seetadet.core.config import cfg
from seetadet.data.assigners import MaxIoUAssigner
from seetadet.ops.normalization import to_tensor
from seetadet.utils.bbox import bbox_overlaps
from seetadet.utils.bbox import bbox_transform
from seetadet.utils.bbox import clip_boxes
from seetadet.utils.bbox import distribute_boxes
from seetadet.utils.bbox import filter_empty_boxes


class ProposalTargets(object):
    """Generate ground-truth targets for proposals."""

    def __init__(self):
        super(ProposalTargets, self).__init__()
        self.num_classes = len(cfg.MODEL.CLASSES)
        self.num_rois = cfg.FAST_RCNN.BATCH_SIZE
        self.num_fg_rois = round(cfg.FAST_RCNN.POSITIVE_FRACTION * self.num_rois)
        self.bbox_reg_weights = cfg.FAST_RCNN.BBOX_REG_WEIGHTS
        self.bbox_reg_cls_agnostic = cfg.FAST_RCNN.BBOX_REG_CLS_AGNOSTIC
        self.mask_size = (cfg.MASK_RCNN.POOLER_RESOLUTION * 2,) * 2
        self.lvl_min, self.lvl_max = cfg.FAST_RCNN.MIN_LEVEL, cfg.FAST_RCNN.MAX_LEVEL
        self.assigner = MaxIoUAssigner(pos_iou_thr=cfg.FAST_RCNN.POSITIVE_OVERLAP,
                                       neg_iou_thr=cfg.FAST_RCNN.NEGATIVE_OVERLAP,
                                       match_low_quality=False)
        self.defaults = {'rois': np.array([[-1, 0, 0, 1, 1]], 'float32'),
                         'labels': np.array([-1], 'int64'),
                         'bbox_targets': np.zeros((1, 4), 'float32'),
                         'mask_targets': np.full((1,) + self.mask_size, -1, 'float32')}

    def sample_rois(self, rois, gt_boxes):
        """Match and sample positive and negative RoIs."""
        # Assign ground-truth according to the IoU.
        labels = self.assigner.assign(rois[:, 1:5], gt_boxes)
        fg_inds = np.where(labels > 0)[0]
        bg_inds = np.where(labels == 0)[0]

        # Include ground-truth boxes as foreground regions.
        batch_inds = np.full((gt_boxes.shape[0], 1), rois[0, 0])
        gt_inds = np.arange(len(rois), len(rois) + len(batch_inds))
        fg_inds = np.concatenate((fg_inds, gt_inds))
        rois = np.vstack((rois, np.hstack((batch_inds, gt_boxes[:, :4]))))

        # Sample foreground regions without replacement.
        num_fg_rois = int(min(self.num_fg_rois, fg_inds.size))
        fg_inds = npr.choice(fg_inds, num_fg_rois, False)

        # Sample background regions without replacement.
        num_bg_rois = self.num_rois - num_fg_rois
        num_bg_rois = min(num_bg_rois, bg_inds.size)
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, num_bg_rois, False)

        # Take values via sampled indices.
        keep_inds = np.append(fg_inds, bg_inds)
        rois = rois[keep_inds]
        overlaps = bbox_overlaps(rois[:, 1:5], gt_boxes[:, :4])
        gt_assignments = overlaps.argmax(axis=1)
        labels = gt_boxes[gt_assignments, 4].astype('int64')

        # Reassign background regions.
        labels[num_fg_rois:] = 0

        return rois, labels, gt_assignments

    def distribute_blobs(self, blobs, lvls):
        """Distribute blobs on given levels."""
        outputs = collections.defaultdict(list)
        lvl_inds = [np.where(lvls == (i + self.lvl_min))[0]
                    for i in range(self.lvl_max - self.lvl_min + 1)]
        for inds in lvl_inds:
            for key, blob in blobs.items():
                outputs[key].append(blob[inds] if len(inds) > 0
                                    else self.defaults[key])
        return outputs

    def get_bbox_targets(self, rois, boxes):
        return bbox_transform(rois, boxes, weights=self.bbox_reg_weights)

    def get_mask_targets(self, rois, segms, inds):
        targets = np.full((len(rois),) + self.mask_size, -1, 'float32')
        masks = segms[inds].crop_and_resize(rois[inds], self.mask_size)
        for i, j in enumerate(inds):
            if masks[i] is not None:
                targets[j] = masks[i]
        return targets

    def compute(self, **inputs):
        """Compute proposal targets."""
        blobs = collections.defaultdict(list)
        all_rois = inputs['rois']
        batch_inds = all_rois[:, 0].astype('int32')

        # Compute targets per image.
        for i, gt_boxes in enumerate(inputs['gt_boxes']):
            # Select proposals of this image.
            rois = all_rois[np.where(batch_inds == i)[0]]
            # Filter empty RoIs.
            rois[:, 1:5] = clip_boxes(rois[:, 1:5], inputs['im_info'][i][:2])
            rois = rois[filter_empty_boxes(rois[:, 1:5])]
            # Sample a batch of RoIs for training.
            rois, labels, gt_assignments = self.sample_rois(rois, gt_boxes)
            # Fill blobs.
            blobs['rois'].append(rois)
            blobs['labels'].append(labels)
            blobs['bbox_targets'].append(self.get_bbox_targets(
                rois[:, 1:5], gt_boxes[gt_assignments, :4]))
            if 'gt_segms' in inputs:
                fg_inds = np.where(labels > 0)[0]
                segms = inputs['gt_segms'][i][gt_assignments]
                targets = self.get_mask_targets(rois[:, 1:5], segms, fg_inds)
                blobs['mask_targets'].append(targets)

        # Concat to get the contiguous blobs.
        blobs = dict((k, np.concatenate(blobs[k])) for k in blobs.keys())

        # Distribute blobs by the level of all ROIs.
        lvls = distribute_boxes(blobs['rois'][:, 1:], self.lvl_min, self.lvl_max)
        blobs = self.distribute_blobs(blobs, lvls)

        # Add the targets using foreground ROIs only.
        for lvl in range(self.lvl_max - self.lvl_min + 1):
            inds = np.where(blobs['labels'][lvl] > 0)[0]
            if len(inds) > 0:
                blobs['fg_rois'].append(blobs['rois'][lvl][inds])
                blobs['mask_labels'].append(blobs['labels'][lvl][inds] - 1)
                if 'mask_targets' in blobs:
                    blobs['mask_targets'][lvl] = blobs['mask_targets'][lvl][inds]
            else:
                blobs['fg_rois'].append(self.defaults['rois'])
                blobs['mask_labels'].append(np.array([0], 'int64'))
                if 'mask_targets' in blobs:
                    blobs['mask_targets'][lvl] = self.defaults['mask_targets']

        # Concat to get contiguous blobs along the levels.
        rois, fg_rois = blobs['rois'], blobs['fg_rois']
        blobs = dict((k, np.concatenate(blobs[k])) for k in blobs.keys())

        # Compute class-specific strides.
        bbox_strides = np.arange(len(blobs['rois'])) * (self.num_classes - 1)
        mask_strides = np.arange(len(blobs['fg_rois'])) * (self.num_classes - 1)

        # Select the foreground RoIs for bbox targets.
        fg_inds = np.where(blobs['labels'] > 0)[0]
        if len(fg_inds) == 0:
            # Sample a proposal randomly to avoid memory issue.
            fg_inds = npr.randint(len(blobs['labels']), size=[1])

        outputs = {
            'rois': [to_tensor(rois[i]) for i in range(len(rois))],
            'fg_rois': [to_tensor(fg_rois[i]) for i in range(len(fg_rois))],
            'labels': to_tensor(blobs['labels']), 'proposals': np.concatenate(rois),
            'bbox_inds': to_tensor(fg_inds if self.bbox_reg_cls_agnostic else
                                   (bbox_strides[fg_inds] + (blobs['labels'][fg_inds] - 1))),
            'mask_inds': to_tensor(mask_strides + blobs['mask_labels']),
            'bbox_targets': to_tensor(blobs['bbox_targets'][fg_inds]),
            'bbox_anchors': to_tensor(blobs['rois'][fg_inds, 1:]),
        }

        if 'mask_targets' in blobs:
            outputs['mask_targets'] = to_tensor(blobs['mask_targets'])

        return outputs
