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
"""VGGNet backbone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import BACKBONES
from seetadet.ops.build import build_norm
from seetadet.ops.normalization import L2Norm


class VGGBlock(nn.Module):
    """The VGG block."""

    def __init__(self, dim_in, dim, downsample=None):
        super(VGGBlock, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim, 3, padding=1,
                              bias=not cfg.BACKBONE.NORM)
        self.bn = build_norm(dim, cfg.BACKBONE.NORM)
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(self.bn(self.conv(x)))


class VGG(nn.Module):
    """VGGNet."""

    def __init__(self, depths):
        super(VGG, self).__init__()
        dim_in, dims, blocks = 3, [64, 128, 256, 512, 512], []
        self.out_indices = [v - 1 for v in itertools.accumulate(depths)][1:]
        self.out_dims = dims[1:]
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            downsample = nn.MaxPool2d(2, 2, ceil_mode=True) if i > 0 else None
            blocks.append(VGGBlock(dim_in, dim, downsample))
            for _ in range(depth - 1):
                blocks.append(VGGBlock(dim, dim))
            setattr(self, 'layer%d' % i, nn.Sequential(*blocks[-depth:]))
            dim_in = dim
        self.blocks = blocks
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs


class VGGFCN(VGG):
    """Fully convolutional VGGNet in SSD."""

    def __init__(self, depths):
        super(VGGFCN, self).__init__(depths)
        dim_in, out_index = self.out_dims[-1], self.out_indices[-1]
        self.blocks.append(nn.Sequential(
            nn.MaxPool2d(3, padding=1),
            nn.Conv2d(dim_in, 1024, 3, padding=6, dilation=6),
            nn.ReLU(True)))
        self.blocks.append(nn.Sequential(nn.Conv2d(1024, 1024, 1), nn.ReLU(True)))
        self.layer4.add_module(str(len(self.layer4)), self.blocks[-2])
        self.layer4.add_module(str(len(self.layer4)), self.blocks[-1])
        self.out_dims = [self.out_dims[-2], 1024]  # conv4_3, fc7
        self.out_indices = [self.out_indices[-2], out_index + 2]  # 9, 14
        self.norm = L2Norm(dim_in, init=20.0)

    def forward(self, x):
        outputs = super(VGGFCN, self).forward(x)
        outputs[0] = self.norm(outputs[0])
        return outputs


BACKBONES.register('vgg16', VGG, depths=(2, 2, 3, 3, 3))
BACKBONES.register('vgg16_fcn', VGGFCN, depths=(2, 2, 3, 3, 3))
