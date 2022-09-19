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
"""ResNet backbone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.core.engine.utils import freeze_module
from seetadet.models.build import BACKBONES
from seetadet.ops.build import build_norm


class BasicBlock(nn.Module):
    """The basic resnet block."""

    expansion = 1

    def __init__(self, dim_in, dim, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim, 3, stride, padding=1, bias=False)
        self.bn1 = build_norm(dim, cfg.BACKBONE.NORM)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn2 = build_norm(dim, cfg.BACKBONE.NORM)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.relu(x.add_(shortcut))


class Bottleneck(nn.Module):
    """The bottleneck resnet block."""

    expansion = 4
    groups, width_per_group = 1, 64

    def __init__(self, dim_in, dim, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        width = int(dim * (self.width_per_group / 64.)) * self.groups
        self.conv1 = nn.Conv2d(dim_in, width, 1, bias=False)
        self.bn1 = build_norm(width, cfg.BACKBONE.NORM)
        self.conv2 = nn.Conv2d(width, width, 3, stride, padding=1, bias=False)
        self.bn2 = build_norm(width, cfg.BACKBONE.NORM)
        self.conv3 = nn.Conv2d(width, dim * self.expansion, 1, bias=False)
        self.bn3 = build_norm(dim * self.expansion, cfg.BACKBONE.NORM)
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.relu(x.add_(shortcut))


class ResNet(nn.Module):
    """ResNet class."""

    def __init__(self, block, depths, stride_in_1x1=False):
        super(ResNet, self).__init__()
        dim_in, dims, blocks = 64, [64, 128, 256, 512], []
        self.out_indices = [v - 1 for v in itertools.accumulate(depths)]
        self.out_dims = [dim_in] + [v * block.expansion for v in dims]
        self.conv1 = nn.Conv2d(3, dim_in, 7, 2, padding=3, bias=False)
        self.bn1 = build_norm(dim_in, cfg.BACKBONE.NORM)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        for i, depth, dim in zip(range(4), depths, dims):
            downsample, stride = None, 1 if i == 0 else 2
            if stride != 1 or dim_in != dim * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim * block.expansion, 1, stride, bias=False),
                    build_norm(dim * block.expansion, cfg.BACKBONE.NORM))
            blocks.append(block(dim_in, dim, stride, downsample))
            if isinstance(blocks[-1], Bottleneck) and stride_in_1x1:
                blocks[-1].conv1.stride = (stride, stride)
                blocks[-1].conv2.stride = (1, 1)
            dim_in = dim * block.expansion
            for _ in range(depth - 1):
                blocks.append(block(dim_in, dim))
            setattr(self, 'layer%d' % (i + 1), nn.Sequential(*blocks[-depth:]))
        self.blocks = blocks
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
        num_freeze_stages = cfg.BACKBONE.FREEZE_AT
        if num_freeze_stages > 0:
            self.conv1.apply(freeze_module)
            self.bn1.apply(freeze_module)
        for i in range(num_freeze_stages - 1, 0, -1):
            getattr(self, 'layer%d' % i).apply(freeze_module)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        outputs = [None]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs


class ResNetV1a(ResNet):
    """ResNet with stride in bottleneck 1x1 convolution."""

    def __init__(self, block, depths):
        super(ResNetV1a, self).__init__(block, depths, stride_in_1x1=True)


BACKBONES.register('resnet18', ResNet, block=BasicBlock, depths=[2, 2, 2, 2])
BACKBONES.register('resnet34', ResNet, block=BasicBlock, depths=[3, 4, 6, 3])
BACKBONES.register('resnet50', ResNet, block=Bottleneck, depths=[3, 4, 6, 3])
BACKBONES.register('resnet101', ResNet, block=Bottleneck, depths=[3, 4, 23, 3])
BACKBONES.register('resnet50_v1a', ResNetV1a, block=Bottleneck, depths=[3, 4, 6, 3])
BACKBONES.register('resnet101_v1a', ResNetV1a, block=Bottleneck, depths=[3, 4, 23, 3])
