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
"""MobileNetV2 backbone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import BACKBONES
from seetadet.ops.conv import ConvNorm2d


def make_divisible(v, divisor=8):
    """Return the divisible value."""
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """Invert residual block."""

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.BACKBONE.NORM,
            activation_type='ReLU6')
        self.has_endpoint = stride == 2
        self.apply_shortcut = stride == 1 and dim_in == dim_out
        self.dim = dim = int(round(dim_in * expand_ratio))
        self.conv1 = (conv_module(dim_in, dim, 1)
                      if expand_ratio > 1 else nn.Identity())
        self.conv2 = conv_module(dim, dim, kernel_size, stride, groups=dim)
        self.conv3 = conv_module(dim, dim_out, 1, activation_type='')

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        if self.has_endpoint:
            self.endpoint = x
        x = self.conv2(x)
        x = self.conv3(x)
        if self.apply_shortcut:
            return x.add_(shortcut)
        return x


class MobileNetV2(nn.Module):
    """MobileNetV2 class."""

    def __init__(self, depths, dims, strides, expand_ratios, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.BACKBONE.NORM,
            activation_type='ReLU6')
        dims = list(map(lambda x: make_divisible(x * width_mult), dims))
        self.conv1 = conv_module(3, dims[0], 3, 2)
        dim_in, blocks = dims[0], []
        self.out_indices, self.out_dims = [], []
        for i, (depth, dim) in enumerate(zip(depths, dims[1:-1])):
            for j in range(depth):
                stride = strides[i] if j == 0 else 1
                blocks.append(InvertedResidual(
                    dim_in, dim, stride=stride,
                    expand_ratio=expand_ratios[i]))
                if blocks[-1].has_endpoint:
                    self.out_indices.append(len(blocks) - 1)
                    self.out_dims.append(blocks[-1].dim)
                dim_in = dim
            setattr(self, 'layer%d' % (i + 1), nn.Sequential(*blocks[-depth:]))
        self.conv2 = conv_module(dim_in, dims[-1], 1)
        self.blocks = blocks + [self.conv2]
        self.out_dims.append(dims[-1])
        self.out_indices.append(len(self.blocks) - 1)

    def forward(self, x):
        x = self.conv1(x)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outputs.append(blk.__dict__.pop('endpoint', x))
        return outputs


BACKBONES.register(
    'mobilenet_v2', MobileNetV2,
    dims=(32,) + (16, 24, 32, 64, 96, 160, 320) + (1280,),
    depths=(1, 2, 3, 4, 3, 3, 1),
    strides=(1, 2, 2, 2, 1, 2, 1),
    expand_ratios=(1, 6, 6, 6, 6, 6, 6))
