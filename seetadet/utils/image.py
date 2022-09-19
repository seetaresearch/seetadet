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
"""Image utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL.Image
import PIL.ImageEnhance


def im_resize(img, size=None, scale=None, mode='linear'):
    """Resize image by the scale or size."""
    if size is None:
        if not isinstance(scale, (tuple, list)):
            scale = (scale, scale)
        h, w = img.shape[:2]
        size = int(h * scale[0] + .5), int(w * scale[1] + .5)
    else:
        if not isinstance(size, (tuple, list)):
            size = (size, size)
    mode = {'linear': PIL.Image.BILINEAR,
            'nearest': PIL.Image.NEAREST}[mode]
    img = PIL.Image.fromarray(img)
    return np.array(img.resize(size[::-1], mode))


def im_rescale(img, scales, max_size=0, keep_ratio=True):
    """Rescale image to match the detecting scales."""
    im_shape = img.shape
    img_list, img_scales = [], []
    if keep_ratio:
        size_min = np.min(im_shape[:2])
        size_max = np.max(im_shape[:2])
        for target_size in scales:
            im_scale = float(target_size) / float(size_min)
            target_size_max = max_size if max_size > 0 else target_size
            if np.round(im_scale * size_max) > target_size_max:
                im_scale = float(target_size_max) / float(size_max)
            img_list.append(im_resize(img, scale=im_scale))
            img_scales.append((im_scale, im_scale))
    else:
        for target_size in scales:
            h_scale = float(target_size) / im_shape[0]
            w_scale = float(target_size) / im_shape[1]
            img_list.append(im_resize(img, size=target_size))
            img_scales.append((h_scale, w_scale))
    return img_list, img_scales


def color_jitter(img, brightness=None, contrast=None, saturation=None):
    """Distort the color of image."""
    def add_transform(transforms, type, range):
        if range is not None:
            if not isinstance(range, (tuple, list)):
                range = (1. - range, 1. + range)
            transforms.append((type, range))
    transforms = []
    contrast_first = np.random.rand() < 0.5
    add_transform(transforms, PIL.ImageEnhance.Brightness, brightness)
    if contrast_first:
        add_transform(transforms, PIL.ImageEnhance.Contrast, contrast)
    add_transform(transforms, PIL.ImageEnhance.Color, saturation)
    if not contrast_first:
        add_transform(transforms, PIL.ImageEnhance.Contrast, contrast)
    for transform, jitter_range in transforms:
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        img = transform(img)
        img = img.enhance(np.random.uniform(*jitter_range))
    return np.asarray(img)
