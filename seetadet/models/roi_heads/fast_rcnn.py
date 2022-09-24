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
"""Fast R-CNN head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.ops.build import build_loss
from seetadet.ops.conv import ConvNorm2d
from seetadet.ops.vision import RoIPooler
from seetadet.ops.vision import ScaleGradient


class FastRCNNHead(nn.Module):
    """Fast R-CNN head."""

    def __init__(self, in_dims):
        super(FastRCNNHead, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.FAST_RCNN.NORM,
            kernel_size=3, activation_type='ReLU')
        self.output_conv = nn.ModuleList()
        self.output_fc = nn.ModuleList()
        for i in range(cfg.FAST_RCNN.NUM_CONV):
            dim = in_dims[0] if i == 0 else cfg.FAST_RCNN.CONV_HEAD_DIM
            self.output_conv += [conv_module(dim, cfg.FAST_RCNN.CONV_HEAD_DIM)]
        for i in range(cfg.FAST_RCNN.NUM_FC):
            dim = in_dims[0] * cfg.FAST_RCNN.POOLER_RESOLUTION ** 2
            dim = dim if i == 0 else cfg.FAST_RCNN.FC_HEAD_DIM
            self.output_fc += [nn.Sequential(nn.Linear(dim, cfg.FAST_RCNN.FC_HEAD_DIM),
                                             nn.ReLU(inplace=True))]
        self.cls_score = nn.Linear(cfg.FAST_RCNN.FC_HEAD_DIM, len(cfg.MODEL.CLASSES))
        num_classes = 1 if cfg.FAST_RCNN.BBOX_REG_CLS_AGNOSTIC else len(cfg.MODEL.CLASSES) - 1
        self.bbox_pred = nn.Linear(cfg.FAST_RCNN.FC_HEAD_DIM, num_classes * 4)
        self.pooler = RoIPooler(
            pooler_type=cfg.FAST_RCNN.POOLER_TYPE,
            resolution=cfg.FAST_RCNN.POOLER_RESOLUTION,
            sampling_ratio=cfg.FAST_RCNN.POOLER_SAMPLING_RATIO)
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.bbox_loss = build_loss(cfg.FAST_RCNN.BBOX_REG_LOSS_TYPE)
        self.spatial_scales = [1. / (2 ** lvl) for lvl in range(
            cfg.FAST_RCNN.MIN_LEVEL, cfg.FAST_RCNN.MAX_LEVEL + 1)]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)

    def get_outputs(self, inputs):
        x = torch.cat([self.pooler(
            inputs['features'][i], inputs['rois'][i],
            spatial_scale=spatial_scale) for i, spatial_scale
            in enumerate(self.spatial_scales)])
        x = ScaleGradient.apply(x, inputs.pop('grad_scale', 1.0))
        for layer in self.output_conv:
            x = layer(x)
        x = x.flatten_(1)
        for layer in self.output_fc:
            x = layer(x)
        cls_score, bbox_pred = self.cls_score(x), self.bbox_pred(x)
        return {'cls_score': cls_score, 'bbox_pred': bbox_pred}

    def get_losses(self, inputs, targets):
        bbox_pred = inputs['bbox_pred'].reshape_((0, -1, 4))
        bbox_pred = bbox_pred.flatten_(0, 1)[targets['bbox_inds']]
        cls_loss = self.cls_loss(inputs['cls_score'], targets['labels'])
        bbox_loss = self.bbox_loss(bbox_pred, targets['bbox_targets'])
        normalizer = cfg.FAST_RCNN.BATCH_SIZE * cfg.TRAIN.IMS_PER_BATCH
        bbox_loss_weight = cfg.FAST_RCNN.BBOX_REG_LOSS_WEIGHT / normalizer
        bbox_loss = bbox_loss.mul_(bbox_loss_weight)
        return {'cls_loss': cls_loss, 'bbox_loss': bbox_loss}

    def forward(self, inputs, targets=None):
        outputs = self.get_outputs(inputs)
        if self.training:
            logits = {'cls_score': outputs['cls_score'].float(),
                      'bbox_pred': outputs['bbox_pred'].float()}
            outputs = self.get_losses(logits, targets)
            outputs['bbox_pred'] = logits['bbox_pred'].data
            return outputs
        else:
            outputs['cls_score'] = nn.functional.softmax(
                outputs['cls_score'], dim=1, inplace=True)
            return {'rois': torch.cat(inputs['rois']),
                    'cls_score': outputs['cls_score'].float(),
                    'bbox_pred': outputs['bbox_pred'].float()}
