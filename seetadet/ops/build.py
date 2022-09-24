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
"""Build for ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import nn

from seetadet.ops.loss import GIoULoss
from seetadet.ops.loss import L1Loss
from seetadet.ops.loss import SmoothL1Loss
from seetadet.ops.loss import SigmoidFocalLoss
from seetadet.ops.normalization import FrozenBatchNorm2d
from seetadet.ops.normalization import TransposedLayerNorm


def build_loss(loss_type, reduction='sum', **kwargs):
    if isinstance(loss_type, str):
        loss_type = loss_type.lower()
        if loss_type != 'smooth_l1':
            kwargs.pop('beta', None)
        loss_type = {
            'l1': L1Loss,
            'smooth_l1': SmoothL1Loss,
            'giou': GIoULoss,
            'cross_entropy': nn.CrossEntropyLoss,
            'sigmoid_focal': SigmoidFocalLoss,
        }[loss_type]
    return loss_type(reduction=reduction, **kwargs)


def build_norm(dim, norm_type):
    """Build the normalization module."""
    if isinstance(norm_type, str):
        if len(norm_type) == 0:
            return nn.Identity()
        norm_type = {
            'BN': nn.BatchNorm2d,
            'FrozenBN': FrozenBatchNorm2d,
            'SyncBN': nn.SyncBatchNorm,
            'LN': TransposedLayerNorm,
            'GN': lambda c: nn.GroupNorm(32, c),
            'Affine': lambda c: FrozenBatchNorm2d(c, affine=True),
        }[norm_type]
    return norm_type(dim)


def build_activation(activation_type, inplace=False):
    """Build the activation module."""
    if isinstance(activation_type, str):
        if len(activation_type) == 0:
            return nn.Identity()
        activation_type = getattr(nn, activation_type)
    activation = activation_type()
    activation.inplace = inplace
    return activation
