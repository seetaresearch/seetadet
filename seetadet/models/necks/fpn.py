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
"""FPN neck."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import NECKS
from seetadet.ops.conv import ConvNorm2d


@NECKS.register('fpn')
class FPN(nn.Module):
    """FPN to enhance input features."""

    def __init__(self, in_dims):
        super(FPN, self).__init__()
        lateral_conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.FPN.NORM)
        output_conv_module = functools.partial(
            ConvNorm2d, norm_type=cfg.FPN.NORM, conv_type=cfg.FPN.CONV)
        self.dim = cfg.FPN.DIM
        self.min_lvl = cfg.FPN.MIN_LEVEL
        self.max_lvl = cfg.FPN.MAX_LEVEL
        self.fuse_lvl = cfg.FPN.FUSE_LEVEL
        self.highest_lvl = min(self.max_lvl, len(in_dims))
        self.coarsest_stride = cfg.BACKBONE.COARSEST_STRIDE
        self.out_dims = [self.dim] * (self.max_lvl - self.min_lvl + 1)
        self.lateral_conv = nn.ModuleList()
        self.output_conv = nn.ModuleList()
        for dim in in_dims[self.min_lvl - 1:self.highest_lvl]:
            self.lateral_conv += [lateral_conv_module(dim, self.dim, 1)]
            self.output_conv += [output_conv_module(self.dim, self.dim, 3)]
        if 'rcnn' not in cfg.MODEL.TYPE:
            for lvl in range(self.highest_lvl + 1, self.max_lvl + 1):
                dim = in_dims[-1] if lvl == self.highest_lvl + 1 else self.dim
                self.output_conv += [output_conv_module(dim, self.dim, 3, stride=2)]

    def forward(self, features):
        features = features[self.min_lvl - 1:self.highest_lvl]
        laterals = [conv(x) for conv, x in zip(self.lateral_conv, features)]
        for i in range(self.fuse_lvl - self.min_lvl, 0, -1):
            y, x = laterals[i - 1], laterals[i]
            scale = 2 if self.coarsest_stride > 1 else None
            size = None if self.coarsest_stride > 1 else y.shape[2:]
            y += nn.functional.interpolate(x, size, scale)
        outputs = [conv(x) for conv, x in zip(self.output_conv, laterals)]
        if len(self.output_conv) <= len(self.lateral_conv):
            for _ in range(len(outputs), len(self.out_dims)):
                outputs.append(nn.functional.max_pool2d(outputs[-1], 1, stride=2))
        else:
            outputs.append(self.output_conv[len(outputs)](features[-1]))
            for i in range(len(outputs), len(self.out_dims)):
                outputs.append(self.output_conv[i](nn.functional.relu(outputs[-1])))
        return outputs
