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
"""Normalization ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.core.engine.utils import get_device


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where statistics or affine parameters are fixed."""

    def __init__(self, num_features, eps=1e-5, affine=False, inplace=True):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.inplace = inplace and (not affine)
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('weight', torch.ones(num_features))
            self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def extra_repr(self):
        affine_str = '{num_features}, eps={eps}, affine={affine}' \
                     .format(**self.__dict__)
        inplace_str = ', inplace' if self.inplace else ''
        return affine_str + inplace_str

    def forward(self, input):
        return nn.functional.affine(
            input, self.weight, self.bias,
            dim=1, out=input if self.inplace else None)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Fuse the running stats into weight and bias.
        # Note that this behavior will break the original stats
        # into zero means and one stds.
        with torch.no_grad():
            self.running_var.float_().add_(self.eps).sqrt_()
            self.weight.float_().div_(self.running_var)
            self.bias.float_().sub_(self.running_mean.float_() * self.weight)
            self.running_mean.zero_()
            self.running_var.one_().sub_(self.eps)


class TransposedLayerNorm(nn.LayerNorm):
    """LayerNorm with pre-transposed spatial axes."""

    def forward(self, input):
        return nn.functional.layer_norm(
            input.permute(0, 2, 3, 1), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class L2Norm(nn.Module):
    """Parameterized L2 normalize."""

    def __init__(self, num_features, init=20., eps=1e-5):
        super(L2Norm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_features).fill_(init))

    def forward(self, input):
        out = nn.functional.normalize(input, p=2, dim=1, eps=self.eps)
        return nn.functional.affine(out, self.weight, dim=1)


class ToTensor(nn.Module):
    """Convert input to tensor."""

    def __init__(self):
        super(ToTensor, self).__init__()
        self.device = torch.device('cpu')
        self.tensor = torch.ones(1)
        self.normalize = functools.partial(
            nn.functional.channel_norm,
            mean=cfg.MODEL.PIXEL_MEAN,
            std=cfg.MODEL.PIXEL_STD,
            dim=1, dims=(0, 3, 1, 2),
            dtype=cfg.MODEL.PRECISION.lower())

    def _apply(self, fn):
        fn(self.tensor)

    def forward(self, input, normalize=False):
        if input is None:
            return input
        if not isinstance(input, torch.Tensor):
            input = torch.from_numpy(input)
        input = input.to(self.tensor.device)
        if normalize and not input.is_floating_point():
            input = self.normalize(input)
        return input


def to_tensor(input, to_device=True):
    """Convert input to tensor."""
    if input is None:
        return input
    if not isinstance(input, torch.Tensor):
        input = torch.from_numpy(input)
    if to_device:
        input = input.to(device=get_device(cfg.GPU_ID))
    return input
