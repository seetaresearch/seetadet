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
"""SSD head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.data.targets.ssd import AnchorTargets
from seetadet.ops.build import build_loss
from seetadet.ops.conv import ConvNorm2d


class SSDHead(nn.Module):
    """SSD head."""

    def __init__(self, in_dims):
        super(SSDHead, self).__init__()
        self.targets = AnchorTargets()
        self.cls_score = nn.ModuleList()
        self.bbox_pred = nn.ModuleList()
        self.num_classes = len(cfg.MODEL.CLASSES)
        conv_module = nn.Conv2d
        if cfg.FPN.CONV == 'SepConv2d':
            conv_module = functools.partial(ConvNorm2d, conv_type='SepConv2d')
        conv_module = functools.partial(conv_module, kernel_size=3, padding=1)
        for i, dim in enumerate(in_dims):
            num_anchors = self.targets.generator.num_cell_anchors(i)
            self.cls_score.append(conv_module(dim, num_anchors * self.num_classes))
            self.bbox_pred.append(conv_module(dim, num_anchors * 4))
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.bbox_loss = build_loss(cfg.SSD.BBOX_REG_LOSS_TYPE)

    def get_outputs(self, inputs):
        """Return the outputs."""
        features = list(inputs['features'])
        cls_score, bbox_pred = [], []
        for i, x in enumerate(features):
            cls_score.append(self.cls_score[i](x).permute(0, 2, 3, 1).flatten_(1))
            bbox_pred.append(self.bbox_pred[i](x).permute(0, 2, 3, 1).flatten_(1))
        cls_score = torch.cat(cls_score, 1) if len(features) > 1 else cls_score[0]
        bbox_pred = torch.cat(bbox_pred, 1) if len(features) > 1 else bbox_pred[0]
        cls_score = cls_score.reshape_((0, -1, self.num_classes))
        bbox_pred = bbox_pred.reshape_((0, -1, 4))
        return {'cls_score': cls_score, 'bbox_pred': bbox_pred}

    def get_losses(self, inputs, targets):
        """Return the losses."""
        cls_score = inputs['cls_score'].flatten_(0, 1)
        bbox_pred = inputs['bbox_pred'].flatten_(0, 1)
        bbox_pred = bbox_pred[targets['bbox_inds']]
        cls_loss = self.cls_loss(cls_score, targets['labels'])
        bbox_loss = self.bbox_loss(bbox_pred, targets['bbox_targets'],
                                   targets['bbox_anchors'])
        normalizer = targets['bbox_inds'].size(0)
        cls_loss_weight = 1.0 / normalizer
        bbox_loss_weight = cfg.SSD.BBOX_REG_LOSS_WEIGHT / normalizer
        cls_loss = cls_loss.mul_(cls_loss_weight)
        bbox_loss = bbox_loss.mul_(bbox_loss_weight)
        return {'cls_loss': cls_loss, 'bbox_loss': bbox_loss}

    def forward(self, inputs):
        outputs = self.get_outputs(inputs)
        cls_score = outputs['cls_score']
        if self.training:
            cls_score_data = nn.functional.softmax(cls_score.data, dim=2)
            targets = self.targets.compute(cls_score=cls_score_data, **inputs)
            logits = {'cls_score': cls_score.float(),
                      'bbox_pred': outputs['bbox_pred'].float()}
            return self.get_losses(logits, targets)
        else:
            cls_score = nn.functional.softmax(cls_score, dim=2, inplace=True)
            return {'cls_score': cls_score.float(),
                    'bbox_pred': outputs['bbox_pred'].float()}
