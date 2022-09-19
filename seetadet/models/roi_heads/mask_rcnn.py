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
"""Mask R-CNN head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.ops.conv import ConvNorm2d
from seetadet.ops.vision import RoIPooler


class MaskRCNNHead(nn.Module):
    """Mask R-CNN head."""

    def __init__(self, in_dims):
        super(MaskRCNNHead, self).__init__()
        self.dim = cfg.MASK_RCNN.CONV_HEAD_DIM
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.MASK_RCNN.NORM,
            kernel_size=3, activation_type='ReLU')
        self.output_conv = nn.ModuleList()
        for i in range(cfg.MASK_RCNN.NUM_CONV):
            dim = in_dims[0] if i == 0 else self.dim
            self.output_conv += [conv_module(dim, self.dim)]
        self.output_conv += [nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, 2, 2),
            nn.ReLU(True))]
        self.mask_pred = nn.Conv2d(self.dim, len(cfg.MODEL.CLASSES) - 1, 1)
        self.pooler = RoIPooler(
            pooler_type=cfg.MASK_RCNN.POOLER_TYPE,
            resolution=cfg.MASK_RCNN.POOLER_RESOLUTION,
            sampling_ratio=cfg.MASK_RCNN.POOLER_SAMPLING_RATIO)
        self.mask_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.spatial_scales = [1. / (2 ** lvl) for lvl in range(
            cfg.FAST_RCNN.MIN_LEVEL, cfg.FAST_RCNN.MAX_LEVEL + 1)]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mask_pred.weight, std=0.001)

    def get_outputs(self, inputs):
        x = torch.cat([self.pooler(
            inputs['features'][i], inputs['rois'][i],
            spatial_scale=spatial_scale) for i, spatial_scale
            in enumerate(self.spatial_scales)])
        for layer in self.output_conv:
            x = layer(x)
        return {'mask_pred': self.mask_pred(x)}

    def get_losses(self, inputs, targets):
        mask_pred = inputs['mask_pred']
        mask_pred = mask_pred.flatten_(0, 1)[targets['mask_inds']]
        mask_loss = self.mask_loss(mask_pred, targets['mask_targets'])
        return {'mask_loss': mask_loss}

    def forward(self, inputs, targets=None):
        outputs = self.get_outputs(inputs)
        if self.training:
            logits = {'mask_pred': outputs['mask_pred'].float()}
            return self.get_losses(logits, targets)
        else:
            outputs['mask_pred'] = nn.functional.sigmoid(
                outputs['mask_pred'], inplace=True).float()
            return {'mask_pred': outputs['mask_pred']}
