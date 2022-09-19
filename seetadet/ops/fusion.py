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
"""Operator fusions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torch

from seetadet.core.registry import Registry

# Pass to fuse adjacent modules.
FUSIONS = Registry('fusions')


@FUSIONS.register([
    'Conv2d+BatchNorm2d',
    'Conv2d+FrozenBatchNorm2d',
    'Conv2d+SyncBatchNorm',
    'ConvTranspose2d+BatchNorm2d',
    'ConvTranspose2d+FrozenBatchNorm2d',
    'ConvTranspose2d+SyncBatchNorm',
    'DepthwiseConv2d+BatchNorm2d',
    'DepthwiseConv2d+FrozenBatchNorm2d',
    'DepthwiseConv2d+SyncBatchNorm'])
def fuse_conv_bn(conv, bn):
    """Fuse Conv and BatchNorm."""
    with torch.no_grad():
        m = bn.running_mean
        if conv.bias is not None:
            m.sub_(conv.bias.float())
        else:
            delattr(conv, 'bias')
        bn.forward = lambda x: x
        t = bn.weight.div((bn.running_var + bn.eps).sqrt_())
        conv._parameters['bias'] = bn.bias.sub(t * m)
        t_conv_shape = [1, conv.out_channels] if conv.transposed else [0, 1]
        t_conv_shape += [1] * len(conv.kernel_size)
        if conv.weight.dtype == 'float16' and t.dtype == 'float32':
            conv.bias.half_()
            weight = conv.weight.float()
            weight.mul_(t.reshape_(t_conv_shape)).half_()
            conv.weight.copy_(weight)
        else:
            conv.weight.mul_(t.reshape_(t_conv_shape))


def get_fusion(*modules):
    """Return the fusion pass between modules."""
    key = '+'.join(m.__class__.__name__ for m in modules)
    return key, FUSIONS.try_get(key)
