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
"""SSD neck."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import NECKS
from seetadet.ops.conv import ConvNorm2d


class SSDNeck(nn.Module):
    """Feature Pyramid Network."""

    def __init__(self, in_dims, out_dims, kernel_sizes, strides, paddings):
        super(SSDNeck, self).__init__()
        self.out_dims = list(in_dims[-2:]) + list(out_dims)
        dim_in, self.blocks = in_dims[-1], nn.ModuleList()
        conv_module = functools.partial(
            ConvNorm2d, conv_type=cfg.FPN.CONV,
            norm_type=cfg.FPN.NORM, activation_type=cfg.FPN.ACTIVATION)
        for dim, kernel_size, stride, padding in zip(
                out_dims, kernel_sizes, strides, paddings):
            self.blocks.append(conv_module(dim_in, dim // 2, 1))
            self.blocks.append(conv_module(dim // 2, dim, kernel_size, stride, padding))
            dim_in = dim

    def forward(self, features):
        x, outputs = features[-1], features[-2:]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i % 2 > 0:
                outputs.append(x)
        return outputs


NECKS.register(
    'ssd300', SSDNeck,
    out_dims=(512, 256, 256, 256),
    kernel_sizes=(3, 3, 3, 3),
    strides=(2, 2, 1, 1),
    paddings=(1, 1, 0, 0))

NECKS.register(
    'ssd512', SSDNeck,
    out_dims=(512, 256, 256, 256, 256),
    kernel_sizes=(3, 3, 3, 3, 4),
    strides=(2, 2, 2, 2, 1),
    paddings=(1, 1, 1, 1, 1))

NECKS.register(
    'ssdlite', SSDNeck,
    out_dims=(512, 256, 256, 128),
    kernel_sizes=(3, 3, 3, 3),
    strides=(2, 2, 2, 2),
    paddings=(1, 1, 1, 1))
