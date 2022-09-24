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
"""R-CNN detectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import nn
import numpy as np

from seetadet.core.config import cfg
from seetadet.data.targets.rcnn import ProposalTargets
from seetadet.models.build import DETECTORS
from seetadet.models.decoders.rpn import RPNDecoder
from seetadet.models.dense_heads.rpn import RPNHead
from seetadet.models.detectors.detector import Detector
from seetadet.models.roi_heads.fast_rcnn import FastRCNNHead
from seetadet.models.roi_heads.mask_rcnn import MaskRCNNHead
from seetadet.utils.bbox import bbox_transform_inv


@DETECTORS.register('faster_rcnn')
class FasterRCNN(Detector):
    """Faster R-CNN detector."""

    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.rpn_head = RPNHead(self.backbone_dims)
        self.bbox_head = FastRCNNHead(self.backbone_dims)
        self.rpn_decoder = RPNDecoder()
        self.proposal_targets = ProposalTargets()

    def get_outputs(self, inputs):
        """Return the detection outputs."""
        inputs = self.get_inputs(inputs)
        inputs['features'] = self.get_features(inputs)
        inputs['grid_info'] = inputs.pop(
            'grid_info', [x.shape[-2:] for x in inputs['features']])
        outputs = self.rpn_head(inputs)
        inputs['rois'] = self.rpn_decoder({
            'cls_score': outputs.pop('rpn_cls_score'),
            'bbox_pred': outputs.pop('rpn_bbox_pred'),
            'im_info': inputs['im_info'],
            'grid_info': inputs['grid_info']})
        if self.training:
            targets = self.proposal_targets.compute(**inputs)
            inputs['rois'] = targets.pop('rois')
            outputs.update(self.bbox_head(inputs, targets))
        else:
            outputs.update(self.bbox_head(inputs))
        return outputs


@DETECTORS.register('mask_rcnn')
class MaskRCNN(Detector):
    """Mask R-CNN detector."""

    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.rpn_head = RPNHead(self.backbone_dims)
        self.bbox_head = FastRCNNHead(self.backbone_dims)
        self.mask_head = MaskRCNNHead(self.backbone_dims)
        self.rpn_decoder = RPNDecoder()
        self.proposal_targets = ProposalTargets()

    def get_outputs(self, inputs):
        """Return the detection outputs."""
        inputs, outputs = self.get_inputs(inputs), {}
        inputs['features'] = self.get_features(inputs)
        inputs['grid_info'] = inputs.pop(
            'grid_info', [x.shape[-2:] for x in inputs['features']])
        outputs.update(self.rpn_head(inputs))
        inputs['rois'] = self.rpn_decoder({
            'cls_score': outputs.pop('rpn_cls_score'),
            'bbox_pred': outputs.pop('rpn_bbox_pred'),
            'im_info': inputs['im_info'],
            'grid_info': inputs['grid_info']})
        if self.training:
            targets = self.proposal_targets.compute(**inputs)
            inputs['rois'] = targets.pop('rois')
            outputs.update(self.bbox_head(inputs, targets))
            inputs['rois'] = targets.pop('fg_rois')
            outputs.update(self.mask_head(inputs, targets))
        else:
            outputs.update(self.bbox_head(inputs))
            self.outputs = {'features': inputs['features']}
        return outputs


@DETECTORS.register('cascade_rcnn')
class CascadeRCNN(Detector):
    """Cascade R-CNN detector."""

    def __init__(self):
        super(CascadeRCNN, self).__init__()
        self.cascade_ious = cfg.CASCADE_RCNN.POSITIVE_OVERLAP
        self.bbox_reg_weights = cfg.CASCADE_RCNN.BBOX_REG_WEIGHTS
        self.rpn_head = RPNHead(self.backbone_dims)
        self.bbox_heads = nn.ModuleList(FastRCNNHead(self.backbone_dims)
                                        for _ in range(len(self.cascade_ious)))
        if cfg.CASCADE_RCNN.MASK_ON:
            self.mask_head = MaskRCNNHead(self.backbone_dims)
        else:
            self.mask_head = None
        self.rpn_decoder = RPNDecoder()
        self.proposal_targets = ProposalTargets()

    def get_outputs(self, inputs):
        """Return the detection outputs."""
        inputs = self.get_inputs(inputs)
        inputs['features'] = self.get_features(inputs)
        inputs['grid_info'] = inputs.pop(
            'grid_info', [x.shape[-2:] for x in inputs['features']])
        outputs = self.rpn_head(inputs)
        inputs['rois'] = self.rpn_decoder({
            'cls_score': outputs.pop('rpn_cls_score'),
            'bbox_pred': outputs.pop('rpn_bbox_pred'),
            'im_info': inputs['im_info'],
            'grid_info': inputs['grid_info']})
        if self.training:
            assigner = self.proposal_targets.assigner
            outputs['cls_loss'], outputs['bbox_loss'], targets = [], [], {}
            for i, bbox_head in enumerate(self.bbox_heads):
                assigner.pos_iou_thr = assigner.neg_iou_thr = self.cascade_ious[i]
                self.proposal_targets.bbox_reg_weights = self.bbox_reg_weights[i]
                stage_targets = self.proposal_targets.compute(**inputs)
                if self.mask_head is not None and 'gt_segms' in inputs:
                    inputs.pop('gt_segms')
                    for k in ('fg_rois', 'mask_inds', 'mask_targets'):
                        targets[k] = stage_targets.pop(k)
                inputs['rois'] = stage_targets.pop('rois')
                inputs['grad_scale'] = 1. / len(self.bbox_heads)
                bbox_outputs = bbox_head(inputs, stage_targets)
                outputs['cls_loss'].append(bbox_outputs['cls_loss'])
                outputs['bbox_loss'].append(bbox_outputs['bbox_loss'])
                if i < len(self.bbox_heads) - 1:
                    proposals = stage_targets.pop('proposals')
                    boxes = bbox_transform_inv(
                        proposals[:, 1:5], bbox_outputs['bbox_pred'].numpy(),
                        weights=self.bbox_reg_weights[i])
                    inputs['rois'] = np.hstack((proposals[:, :1], boxes))
            if self.mask_head is not None:
                inputs['rois'] = targets.pop('fg_rois')
                outputs.update(self.mask_head(inputs, targets))
        else:
            outputs.update(self.bbox_heads[0](inputs))
            self.outputs = {'features': inputs['features'], 'rois': inputs['rois']}
        return outputs
