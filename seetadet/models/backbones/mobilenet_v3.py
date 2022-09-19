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
"""MobileNetV3 backbone."""

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


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, dim_in, dim):
        super(SqueezeExcite, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim_in, 1)
        self.activation1 = nn.ReLU(True)
        self.activation2 = nn.Hardsigmoid(True)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.activation1(self.conv1(scale))
        scale = self.activation2(self.conv2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """Invert residual block."""

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size=3,
        stride=1,
        expand_ratio=3,
        squeeze_ratio=1,
        activation_type='ReLU',
    ):
        super(InvertedResidual, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.BACKBONE.NORM,
            activation_type=activation_type)
        self.has_endpoint = stride == 2
        self.apply_shortcut = stride == 1 and dim_in == dim_out
        self.dim = dim = int(round(dim_in * expand_ratio))
        self.conv1 = (conv_module(dim_in, dim, 1)
                      if expand_ratio > 1 else nn.Identity())
        self.conv2 = conv_module(dim, dim, kernel_size, stride, groups=dim)
        self.se = (SqueezeExcite(dim, make_divisible(dim * squeeze_ratio))
                   if squeeze_ratio < 1 else nn.Identity())
        self.conv3 = conv_module(dim, dim_out, 1, activation_type='')

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        if self.has_endpoint:
            self.endpoint = x
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.apply_shortcut:
            return x.add_(shortcut)
        return x


class MobileNetV3(nn.Module):
    """MobileNetV3 class."""

    def __init__(self, depths, dims, kernel_sizes, strides,
                 expand_ratios, squeeze_ratios, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.BACKBONE.NORM,
            activation_type='Hardswish')
        dims = list(map(lambda x: make_divisible(x * width_mult), dims))
        self.conv1 = conv_module(3, dims[0], 3, 2)
        dim_in, blocks, coarsest_stride = dims[0], [], 2
        self.out_indices, self.out_dims = [], []
        for i, (depth, dim) in enumerate(zip(depths, dims[1:])):
            coarsest_stride *= strides[i]
            layer_expand_ratios = expand_ratios[i]
            if not isinstance(layer_expand_ratios, (tuple, list)):
                layer_expand_ratios = [layer_expand_ratios]
            layer_expand_ratios = list(layer_expand_ratios)
            layer_expand_ratios += ([layer_expand_ratios[-1]] *
                                    (depth - len(layer_expand_ratios)))
            for j in range(depth):
                blocks.append(InvertedResidual(
                    dim_in, dim,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i] if j == 0 else 1,
                    expand_ratio=layer_expand_ratios[j],
                    squeeze_ratio=squeeze_ratios[i],
                    activation_type='Hardswish'
                    if coarsest_stride >= 16 else 'ReLU'))
                if blocks[-1].has_endpoint:
                    self.out_indices.append(len(blocks) - 1)
                    self.out_dims.append(blocks[-1].dim)
                dim_in = dim
            setattr(self, 'layer%d' % (i + 1), nn.Sequential(*blocks[-depth:]))
        self.conv2 = conv_module(dim_in, blocks[-1].dim, 1)
        self.blocks = blocks + [self.conv2]
        self.out_dims.append(blocks[-1].dim)
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
    'mobilenet_v3_large', MobileNetV3,
    dims=(16,) + (16, 24, 40, 80, 112, 160),
    depths=(1, 2, 3, 4, 2, 3),
    kernel_sizes=(3, 3, 5, 3, 3, 5),
    strides=(1, 2, 2, 2, 1, 2),
    expand_ratios=(1, (4, 3), 3, (6, 2.5, 2.3, 2.3), 6, 6),
    squeeze_ratios=(1, 1, 0.25, 1, 0.25, 0.25))

BACKBONES.register(
    'mobilenet_v3_small', MobileNetV3,
    dims=(16,) + (16, 24, 40, 48, 96),
    depths=(1, 2, 3, 2, 3),
    kernel_sizes=(3, 3, 5, 5, 5),
    strides=(2, 2, 2, 1, 2),
    expand_ratios=(1, (4.5, 88. / 24), (4, 6, 6), 3, 6),
    squeeze_ratios=(0.25, 1, 0.25, 0.25, 0.25))
