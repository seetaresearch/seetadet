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
"""Loss ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg


class GIoULoss(nn.Module):
    """GIoU loss."""

    def __init__(self, reduction='sum', delta_weights=None):
        super(GIoULoss, self).__init__()
        self.reduction = reduction
        self.delta_weights = delta_weights

    def transform_inv(self, boxes, deltas):
        widths = boxes[:, 2:3] - boxes[:, 0:1]
        heights = boxes[:, 3:4] - boxes[:, 1:2]
        ctr_x = boxes[:, 0:1] + 0.5 * widths
        ctr_y = boxes[:, 1:2] + 0.5 * heights
        dx, dy, dw, dh = torch.chunk(deltas, chunks=4, dim=1)
        if self.delta_weights is not None:
            wx, wy, ww, wh = self.delta_weights
            dx, dy, dw, dh = dx / wx, dy / wy, dw / ww, dh / wh
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        return x1, y1, x2, y2

    def forward_impl(self, input, target, anchor):
        x1, y1, x2, y2 = self.transform_inv(anchor, input)
        x1_, y1_, x2_, y2_ = self.transform_inv(anchor, target)
        # Compute the independent area.
        pred_area = (x2 - x1) * (y2 - y1)
        target_area = (x2_ - x1_) * (y2_ - y1_)
        # Compute the intersecting area.
        x1_inter = torch.maximum(x1, x1_)
        y1_inter = torch.maximum(y1, y1_)
        x2_inter = torch.minimum(x2, x2_)
        y2_inter = torch.minimum(y2, y2_)
        w_inter = torch.clamp(x2_inter - x1_inter, min=0)
        h_inter = torch.clamp(y2_inter - y1_inter, min=0)
        area_inter = w_inter * h_inter
        # Compute the enclosing area.
        x1_enc = torch.minimum(x1, x1_)
        y1_enc = torch.minimum(y1, y1_)
        x2_enc = torch.maximum(x2, x2_)
        y2_enc = torch.maximum(y2, y2_)
        area_enc = (x2_enc - x1_enc) * (y2_enc - y1_enc) + 1.
        # Compute the differentiable IoU metric.
        area_union = pred_area + target_area - area_inter
        iou = area_inter / (area_union + 1.)
        iou_metric = iou - (area_enc - area_union) / area_enc
        # Compute the reduced loss.
        if self.reduction == 'sum':
            return (1 - iou_metric).sum()
        else:
            return (1 - iou_metric).mean()

    def forward(self, *inputs, **kwargs):
        with dragon.variable_scope('IoULossVariable'):
            return self.forward_impl(*inputs, **kwargs)


class L1Loss(nn.L1Loss):
    """L1 loss."""

    def forward(self, input, target, *args):
        return super(L1Loss, self).forward(input, target)


class SigmoidFocalLoss(nn.SigmoidFocalLoss):
    """Sigmoid focal loss."""

    def __init__(self, reduction='sum'):
        super(SigmoidFocalLoss, self).__init__(
            alpha=cfg.MODEL.FOCAL_LOSS_ALPHA,
            gamma=cfg.MODEL.FOCAL_LOSS_GAMMA,
            start_index=1,  # Foreground index
            reduction=reduction)


class SmoothL1Loss(nn.SmoothL1Loss):
    """Smoothed l1 loss."""

    def forward(self, input, target, *args):
        return nn.functional.smooth_l1_loss(
            input, target, beta=self.beta,
            reduction=self.reduction)
