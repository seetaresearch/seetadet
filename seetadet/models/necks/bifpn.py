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
"""BiFPN neck."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import NECKS
from seetadet.ops.build import build_activation
from seetadet.ops.conv import ConvNorm2d


class FuseOp(nn.Module):
    """Operator to fuse input features."""

    def __init__(self, num_inputs):
        super(FuseOp, self).__init__()
        self.fuse_type = cfg.FPN.FUSE_TYPE
        if self.fuse_type == 'weighted':
            self.weight = nn.Parameter(torch.ones(num_inputs))

    def forward(self, *inputs):
        if self.fuse_type == 'weighted':
            weights = nn.functional.softmax(self.weight, dim=0).split(1)
            outputs = inputs[0] * weights[0]
            for x, w in zip(inputs[1:], weights[1:]):
                outputs += x * w
        else:
            outputs = inputs[0]
            for x in inputs[1:]:
                outputs += x
        return outputs


class Block(nn.Module):
    """BiFPN block."""

    def __init__(self, in_dims=None):
        super(Block, self).__init__()
        conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.FPN.NORM, conv_type=cfg.FPN.CONV)
        self.dim = cfg.FPN.DIM
        self.min_lvl = cfg.FPN.MIN_LEVEL
        self.max_lvl = cfg.FPN.MAX_LEVEL
        self.highest_lvl = min(self.max_lvl, len(in_dims))
        self.coarsest_stride = cfg.BACKBONE.COARSEST_STRIDE
        self.output_conv1, self.output_fuse1 = nn.ModuleList(), nn.ModuleList()
        self.output_conv2, self.output_fuse2 = nn.ModuleList(), nn.ModuleList()
        for lvl in range(self.min_lvl, self.max_lvl):
            self.output_conv1 += [conv_module(self.dim, self.dim, 3)]
            self.output_conv2 += [conv_module(self.dim, self.dim, 3)]
            self.output_fuse1 += [FuseOp(2)]
            self.output_fuse2 += [FuseOp(3 if lvl < self.max_lvl - 1 else 2)]
        self.activation = build_activation(cfg.FPN.ACTIVATION, inplace=True)

    def forward(self, laterals1, laterals2=None):
        outputs = [laterals1[-1]]
        for i in range(len(laterals1) - 1, 0, -1):
            x1, x2 = outputs[0], laterals1[i - 1]
            scale = 2 if self.coarsest_stride > 1 else None
            size = None if self.coarsest_stride > 1 else x2.shape[2:]
            x1 = nn.functional.interpolate(x1, size, scale)
            y = self.output_fuse1[i - 1](x1, x2)
            outputs.insert(0, self.output_conv1[i - 1](self.activation(y)))
        if laterals2 is None:
            laterals2 = laterals1[1:]
        else:
            laterals2 += laterals1[self.highest_lvl - self.min_lvl + 1:]
        for i in range(1, len(outputs)):
            x1, x2 = outputs[i - 1], laterals2[i - 1]
            x1 = nn.functional.max_pool2d(x1, 3, 2, padding=1)
            if i < len(outputs) - 1:
                y = self.output_fuse2[i - 1](x1, x2, outputs[i])
            else:
                y = self.output_fuse2[i - 1](x1, x2)
            outputs[i] = self.output_conv2[i - 1](self.activation(y))
        return outputs


@NECKS.register('bifpn')
class BiFPN(nn.Module):
    """BiFPN to enhance input features."""

    def __init__(self, in_dims=None):
        super(BiFPN, self).__init__()
        conv_module = functools.partial(ConvNorm2d, norm_type=cfg.FPN.NORM)
        self.dim = cfg.FPN.DIM
        self.min_lvl = cfg.FPN.MIN_LEVEL
        self.max_lvl = cfg.FPN.MAX_LEVEL
        self.highest_lvl = min(self.max_lvl, len(in_dims))
        self.out_dims = [self.dim] * (self.max_lvl - self.min_lvl + 1)
        self.lateral_conv1 = nn.ModuleList()
        self.lateral_conv2 = nn.ModuleList()
        for dim in in_dims[self.min_lvl - 1:self.highest_lvl]:
            self.lateral_conv1 += [conv_module(dim, self.dim, 1)]
        for dim in in_dims[self.min_lvl:self.highest_lvl]:
            self.lateral_conv2 += [conv_module(dim, self.dim, 1)]
        for lvl in range(self.highest_lvl + 1, self.max_lvl + 1):
            dim = in_dims[-1] if lvl == self.highest_lvl + 1 else self.dim
            self.lateral_conv1 += [conv_module(dim, self.dim, 1)
                                   if lvl == self.highest_lvl + 1 else nn.Identity()]
        self.blocks = nn.ModuleList(Block(in_dims) for _ in range(cfg.FPN.NUM_BLOCKS))

    def forward(self, features):
        features = features[self.min_lvl - 1:self.highest_lvl]
        laterals1 = [conv(x) for conv, x in zip(self.lateral_conv1, features)]
        laterals2 = [conv(x) for conv, x in zip(self.lateral_conv2, features[1:])]
        x = features[-1]
        for i in range(len(laterals1), len(self.out_dims)):
            x = self.lateral_conv1[i](x)
            x = nn.functional.max_pool2d(x, 3, 2, padding=1)
            laterals1.append(x)
        for i, blk in enumerate(self.blocks):
            laterals1 = blk(laterals1, laterals2 if i == 0 else None)
        return laterals1
