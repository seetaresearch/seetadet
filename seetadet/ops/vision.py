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
"""Vision ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torchvision
from dragon.vm.torch import nn
from dragon.vm.torch import autograd


class RoIPooler(nn.Module):
    """Resample RoI features into a fixed resolution."""

    def __init__(self, pooler_type='RoIAlign', resolution=7, sampling_ratio=0):
        super(RoIPooler, self).__init__()
        if not isinstance(resolution, (tuple, list)):
            resolution = (resolution, resolution)
        self.pooler_type = pooler_type
        self.resolution = resolution
        self.sampling_ratio = sampling_ratio

    def forward(self, input, boxes, spatial_scale=1.0):
        if self.pooler_type == 'RoIPool':
            return torchvision.ops.roi_pool(
                input, boxes,
                output_size=self.resolution,
                spatial_scale=spatial_scale)
        elif self.pooler_type == 'RoIAlign':
            return torchvision.ops.roi_align(
                input, boxes,
                output_size=self.resolution,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=False)
        elif self.pooler_type == 'RoIAlignV2':
            return torchvision.ops.roi_align(
                input, boxes,
                output_size=self.resolution,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True)
        else:
            raise NotImplementedError


class NonMaxSuppression(object):
    """Filter out boxes that have high IoU with selected ones."""

    @staticmethod
    def apply(input, iou_threshold=0.5):
        return autograd.Function.apply(
            'NonMaxSuppression', input.device, [input],
            iou_threshold=float(iou_threshold))

    autograd.Function.register(
        'NonMaxSuppression', lambda **kwargs: {
            'iou_threshold': kwargs.get('iou_threshold', 0.5),
        })


class PasteMask(object):
    """Paste a set of masks on an image."""

    @staticmethod
    def apply(masks, boxes, output_size, mask_threshold=0.5):
        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
        return autograd.Function.apply(
            'PasteMask', masks.device, [masks, boxes],
            mask_threshold=float(mask_threshold),
            num_sizes=len(output_size), sizes=output_size)

    autograd.Function.register(
        'PasteMask', lambda **kwargs: {
            'mask_threshold': kwargs.get('mask_threshold', 0.5),
            'sizes_desc': 'int64',
        })


class ScaleGradient(object):
    """Scale the graidnet of input tensors."""

    @staticmethod
    def apply(inputs, scale=1.0):
        if scale == 1.0:
            return inputs
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        return autograd.Function.apply(
            'IdentityScale', inputs[0].device, inputs, scale=float(scale))

    autograd.Function.register(
        'IdentityScale', lambda **kwargs: {
            'scale': kwargs.get('scale', 1.0),
        })
