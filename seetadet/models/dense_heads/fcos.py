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
"""FCOS head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.data.targets.fcos import AnchorTargets
from seetadet.models.dense_heads.retinanet import RetinaNetHead
from seetadet.ops.build import build_activation
from seetadet.ops.build import build_loss
from seetadet.ops.build import build_norm
from seetadet.ops.conv import ConvNorm2d


class FCOSHead(RetinaNetHead):
    """FCOS head."""

    def __init__(self, in_dims):
        super(FCOSHead, self).__init__(in_dims)
        conv_module = functools.partial(
            ConvNorm2d, dim_in=in_dims[0], dim_out=in_dims[0],
            kernel_size=3, conv_type=cfg.FCOS.CONV)
        norm_module = functools.partial(build_norm, norm_type=cfg.FCOS.NORM)
        self.conv_module = conv_module
        self.dim_cls = len(cfg.MODEL.CLASSES) - 1
        self.cls_conv = nn.ModuleList(
            conv_module() for _ in range(cfg.FCOS.NUM_CONV))
        self.bbox_conv = nn.ModuleList(
            conv_module() for _ in range(cfg.FCOS.NUM_CONV))
        self.cls_norm = nn.ModuleList()
        self.bbox_norm = nn.ModuleList()
        for _ in range(len(self.cls_conv)):
            for m in (self.cls_norm, self.bbox_norm):
                layer = norm_module(in_dims[0])
                if 'BN' in cfg.FCOS.NORM:
                    layer = nn.ModuleList([layer] + [
                        norm_module(in_dims[0]) for _ in range(len(in_dims) - 1)])
                m.append(layer)
        self.targets = AnchorTargets()
        num_anchors = self.targets.generator.num_cell_anchors(0)
        self.cls_score = conv_module(dim_out=self.dim_cls * num_anchors)
        self.bbox_pred = conv_module(dim_out=4 * num_anchors)
        self.ctrness = conv_module(dim_out=1)
        self.activation = build_activation(cfg.FCOS.ACTIVATION, inplace=True)
        self.cls_loss = build_loss('sigmoid_focal')
        self.bbox_loss = build_loss(cfg.FCOS.BBOX_REG_LOSS_TYPE,
                                    transform_type='linear', reduction='none')
        self.ctrness_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.reset_parameters()

    def get_outputs(self, inputs):
        """Return the outputs."""
        features = list(inputs['features'])
        cls_score, bbox_pred, ctrness = [], [], []
        for j, feature in enumerate(features):
            cls_input, box_input = feature, feature
            for i in range(len(self.cls_conv)):
                cls_conv, box_conv = self.cls_conv[i], self.bbox_conv[i]
                cls_norm, box_norm = self.cls_norm[i], self.bbox_norm[i]
                if isinstance(cls_conv, nn.ModuleList):
                    cls_conv, box_conv = cls_conv[j], box_conv[j]
                if isinstance(cls_norm, nn.ModuleList):
                    cls_norm, box_norm = cls_norm[j], box_norm[j]
                cls_input, box_input = cls_conv(cls_input), box_conv(box_input)
                cls_input = self.activation(cls_norm(cls_input))
                box_input = self.activation(box_norm(box_input))
            cls_score.append(self.cls_score(cls_input).reshape_((0, self.dim_cls, -1)))
            bbox_pred.append(self.bbox_pred(box_input).reshape_((0, 4, -1)))
            ctrness.append(self.ctrness(box_input).reshape_((0, 1, -1)))
        cls_score = torch.cat(cls_score, 2) if len(features) > 1 else cls_score[0]
        bbox_pred = torch.cat(bbox_pred, 2) if len(features) > 1 else bbox_pred[0]
        ctrness = torch.cat(ctrness, 2) if len(features) > 1 else ctrness[0]
        return {'cls_score': cls_score, 'bbox_pred': bbox_pred, 'ctrness': ctrness}

    def get_losses(self, inputs, targets):
        """Return the losses."""
        bbox_pred = inputs['bbox_pred'].permute(0, 2, 1)
        bbox_pred = bbox_pred.flatten_(0, 1)[targets['bbox_inds']]
        ctrness = inputs['ctrness'].flatten_()[targets['bbox_inds']]
        cls_loss = self.cls_loss(inputs['cls_score'], targets['labels'])
        bbox_loss = self.bbox_loss(bbox_pred, targets['bbox_targets'],
                                   targets['bbox_anchors'])
        bbox_loss = bbox_loss.mul_(targets['ctrness_targets']).sum()
        ctrness_loss = self.ctrness_loss(ctrness, targets['ctrness_targets'])
        num_pos, sum_ctrness = torch.distributed.all_reduce(
            torch.tensor([targets['bbox_inds'].size(0),
                          targets['ctrness_targets'].sum().item()],
                         dtype=torch.float32), op='mean').tolist()
        cls_loss_weight = 1.0 / num_pos
        bbox_loss_weight = cfg.FCOS.BBOX_REG_LOSS_WEIGHT / sum_ctrness
        cls_loss = cls_loss.mul_(cls_loss_weight)
        bbox_loss = bbox_loss.mul_(bbox_loss_weight)
        ctrness_loss = ctrness_loss.mul_(cls_loss_weight)
        return {'cls_loss': cls_loss, 'bbox_loss': bbox_loss,
                'ctrness_loss': ctrness_loss}

    def forward(self, inputs):
        outputs = self.get_outputs(inputs)
        if self.training:
            targets = self.targets.compute(**inputs)
            logits = {'cls_score': outputs['cls_score'].float(),
                      'bbox_pred': outputs['bbox_pred'].float(),
                      'ctrness': outputs['ctrness'].float()}
            return self.get_losses(logits, targets)
        else:
            cls_score = nn.functional.sigmoid(outputs['cls_score'], inplace=True)
            cls_score *= nn.functional.sigmoid(outputs['ctrness'], inplace=True)
            return {'cls_score': cls_score.sqrt_().permute(0, 2, 1).float(),
                    'bbox_pred': outputs['bbox_pred'].float()}
