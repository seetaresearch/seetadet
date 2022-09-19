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
"""RPN head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.data.targets.rpn import AnchorTargets
from seetadet.ops.build import build_loss


class RPNHead(nn.Module):
    """RPN head."""

    def __init__(self, in_dims):
        super(RPNHead, self).__init__()
        self.targets = AnchorTargets()
        dim, num_anchors = in_dims[0], self.targets.generator.num_cell_anchors(0)
        self.output_conv = nn.ModuleList(nn.Conv2d(
            dim, dim, 3, padding=1) for _ in range(cfg.RPN.NUM_CONV))
        self.cls_score = nn.Conv2d(dim, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(dim, num_anchors * 4, 1)
        self.activation = nn.ReLU(inplace=True)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.bbox_loss = build_loss(cfg.RPN.BBOX_REG_LOSS_TYPE, beta=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

    def get_outputs(self, inputs):
        """Return the outputs."""
        features = list(inputs['features'])
        cls_score, bbox_pred = [], []
        for x in features:
            for conv in self.output_conv:
                x = self.activation(conv(x))
            cls_score.append(self.cls_score(x).reshape_((0, -1)))
            bbox_pred.append(self.bbox_pred(x).reshape_((0, 4, -1)))
        cls_score = torch.cat(cls_score, 1) if len(features) > 1 else cls_score[0]
        bbox_pred = torch.cat(bbox_pred, 2) if len(features) > 1 else bbox_pred[0]
        return {'rpn_cls_score': cls_score, 'rpn_bbox_pred': bbox_pred}

    def get_losses(self, inputs, targets):
        """Return the losses."""
        bbox_pred = inputs['bbox_pred'].permute(0, 2, 1)
        bbox_pred = bbox_pred.flatten_(0, 1)[targets['bbox_inds']]
        cls_score = inputs['cls_score'].flatten(0, 1)[targets['cls_inds']]
        cls_loss = self.cls_loss(cls_score, targets['labels'])
        bbox_loss = self.bbox_loss(bbox_pred, targets['bbox_targets'],
                                   targets['bbox_anchors'])
        normalizer = cfg.RPN.BATCH_SIZE * cfg.TRAIN.IMS_PER_BATCH
        bbox_loss_weight = cfg.RPN.BBOX_REG_LOSS_WEIGHT / normalizer
        bbox_loss = bbox_loss.mul_(bbox_loss_weight)
        return {'rpn_cls_loss': cls_loss, 'rpn_bbox_loss': bbox_loss}

    def forward(self, inputs):
        outputs = self.get_outputs(inputs)
        outputs['rpn_bbox_pred'] = outputs['rpn_bbox_pred'].float()
        outputs['rpn_cls_score'] = outputs['rpn_cls_score'].float()
        if self.training:
            targets = self.targets.compute(**inputs)
            rpn_cls_score = outputs.pop('rpn_cls_score')
            outputs['rpn_cls_score'] = rpn_cls_score.data
            logits = {'cls_score': rpn_cls_score,
                      'bbox_pred': outputs['rpn_bbox_pred']}
            outputs.update(self.get_losses(logits, targets))
        return outputs
