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
"""Convolution ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import nn

from seetadet.ops.build import build_norm


class ConvNorm2d(nn.Sequential):
    """2d convolution followed by norm."""

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        conv_type='Conv2d',
        norm_type='',
        activation_type='',
        inplace=True,
    ):
        super(ConvNorm2d, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        if conv_type == 'Conv2d':
            layers = [nn.Conv2d(dim_in, dim_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias and (not norm_type))]
        elif conv_type == 'SepConv2d':
            layers = [nn.Conv2d(dim_in, dim_in,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=dim_in,
                                bias=False),
                      nn.Conv2d(dim_in, dim_out,
                                kernel_size=1,
                                bias=bias and (not norm_type))]
        else:
            raise ValueError('Unknown conv type: ' + conv_type)
        if norm_type:
            layers += [build_norm(dim_out, norm_type)]
        if activation_type:
            layers += [getattr(nn, activation_type)()]
            layers[-1].inplace = inplace
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
