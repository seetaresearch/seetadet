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
"""RetinaNet head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.data.targets.retinanet import AnchorTargets
from seetadet.ops.build import build_activation
from seetadet.ops.build import build_loss
from seetadet.ops.build import build_norm
from seetadet.ops.conv import ConvNorm2d
from seetadet.ops.fusion import fuse_conv_bn
from seetadet.utils import profiler


class RetinaNetHead(nn.Module):
    """RetinaNet head."""

    def __init__(self, in_dims):
        super(RetinaNetHead, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, dim_in=in_dims[0], dim_out=in_dims[0],
            kernel_size=3, conv_type=cfg.RETINANET.CONV)
        norm_module = functools.partial(build_norm, norm_type=cfg.RETINANET.NORM)
        self.conv_module = conv_module
        self.dim_cls = len(cfg.MODEL.CLASSES) - 1
        self.cls_conv = nn.ModuleList(
            conv_module() for _ in range(cfg.RETINANET.NUM_CONV))
        self.bbox_conv = nn.ModuleList(
            conv_module() for _ in range(cfg.RETINANET.NUM_CONV))
        self.cls_norm = nn.ModuleList()
        self.bbox_norm = nn.ModuleList()
        for _ in range(len(self.cls_conv)):
            self.cls_norm.append(nn.ModuleList())
            self.bbox_norm.append(nn.ModuleList())
            for _ in range(len(in_dims)):
                self.cls_norm[-1].append(norm_module(in_dims[0]))
                self.bbox_norm[-1].append(norm_module(in_dims[0]))
        self.targets = AnchorTargets()
        num_anchors = self.targets.generator.num_cell_anchors(0)
        self.cls_score = conv_module(dim_out=self.dim_cls * num_anchors)
        self.bbox_pred = conv_module(dim_out=4 * num_anchors)
        self.activation = build_activation(cfg.RETINANET.ACTIVATION, inplace=True)
        self.cls_loss = build_loss('sigmoid_focal')
        self.bbox_loss = build_loss(cfg.RETINANET.BBOX_REG_LOSS_TYPE, beta=0.1)
        self.ema_normalizer = profiler.ExponentialMovingAverage()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
        # Bias prior initialization for focal loss.
        for name, param in self.cls_score.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(param, -math.log((1 - 0.01) / 0.01))

    def optimize_for_inference(self):
        """Optimize modules for inference."""
        if hasattr(self.cls_norm[0][0], 'momentum'):
            cls_conv = nn.ModuleList()
            bbox_conv = nn.ModuleList()
            for i in range(len(self.cls_norm)):
                cls_conv.append(nn.ModuleList())
                bbox_conv.append(nn.ModuleList())
                cls_state = self.cls_conv[i].state_dict()
                bbox_state = self.bbox_conv[i].state_dict()
                for j in range(len(self.cls_norm[i])):
                    cls_conv[i].append(self.conv_module()._apply(
                        lambda t: t.to(self.cls_norm[i][j].weight.device)))
                    bbox_conv[i].append(self.conv_module()._apply(
                        lambda t: t.to(self.bbox_norm[i][j].weight.device)))
                    cls_conv[i][j].load_state_dict(cls_state)
                    bbox_conv[i][j].load_state_dict(bbox_state)
                    fuse_conv_bn(cls_conv[i][j][-1], self.cls_norm[i][j])
                    fuse_conv_bn(bbox_conv[i][j][-1], self.bbox_norm[i][j])
            self._modules['cls_conv'] = cls_conv
            self._modules['bbox_conv'] = bbox_conv

    def get_outputs(self, inputs):
        """Return the outputs."""
        features = list(inputs['features'])
        cls_score, bbox_pred = [], []
        for j, feature in enumerate(features):
            cls_input, box_input = feature, feature
            for i in range(len(self.cls_conv)):
                if isinstance(self.cls_conv[i], nn.ModuleList):
                    cls_input = self.cls_conv[i][j](cls_input)
                    box_input = self.bbox_conv[i][j](box_input)
                else:
                    cls_input = self.cls_conv[i](cls_input)
                    box_input = self.bbox_conv[i](box_input)
                cls_input = self.activation(self.cls_norm[i][j](cls_input))
                box_input = self.activation(self.bbox_norm[i][j](box_input))
            cls_score.append(self.cls_score(cls_input).reshape_((0, self.dim_cls, -1)))
            bbox_pred.append(self.bbox_pred(box_input).reshape_((0, 4, -1)))
        cls_score = torch.cat(cls_score, 2) if len(features) > 1 else cls_score[0]
        bbox_pred = torch.cat(bbox_pred, 2) if len(features) > 1 else bbox_pred[0]
        return {'cls_score': cls_score, 'bbox_pred': bbox_pred}

    def get_losses(self, inputs, targets):
        """Return the losses."""
        bbox_pred = inputs['bbox_pred'].permute(0, 2, 1)
        bbox_pred = bbox_pred.flatten_(0, 1)[targets['bbox_inds']]
        cls_loss = self.cls_loss(inputs['cls_score'], targets['labels'])
        bbox_loss = self.bbox_loss(bbox_pred, targets['bbox_targets'],
                                   targets['bbox_anchors'])
        normalizer = self.ema_normalizer.update(targets['bbox_inds'].size(0))
        cls_loss_weight = 1.0 / normalizer
        bbox_loss_weight = cfg.RETINANET.BBOX_REG_LOSS_WEIGHT / normalizer
        cls_loss = cls_loss.mul_(cls_loss_weight)
        bbox_loss = bbox_loss.mul_(bbox_loss_weight)
        return {'cls_loss': cls_loss, 'bbox_loss': bbox_loss}

    def forward(self, inputs):
        outputs = self.get_outputs(inputs)
        if self.training:
            targets = self.targets.compute(**inputs)
            logits = {'cls_score': outputs['cls_score'].float(),
                      'bbox_pred': outputs['bbox_pred'].float()}
            return self.get_losses(logits, targets)
        else:
            cls_score = outputs['cls_score'].permute(0, 2, 1)
            cls_score = nn.functional.sigmoid(cls_score, inplace=True)
            return {'cls_score': cls_score.float(),
                    'bbox_pred': outputs['bbox_pred'].float()}
