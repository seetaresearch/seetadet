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
"""Mask structure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import numpy as np

from seetadet.utils.polygon import crop_polygons
from seetadet.utils.polygon import flip_polygons
from seetadet.utils.mask import mask_from


class PolygonMasks(object):
    """Polygon masks."""

    def __init__(self, shape=None):
        self.data = []
        self.shape = list(shape)

    def new_masks(self, data, copy=False):
        """Return a new masks object."""
        ret = PolygonMasks(self.shape)
        ret.data = deepcopy(data) if copy else data
        return ret

    def apply_flip(self):
        """Apply flip transform."""
        for i, mask in enumerate(self.data):
            if mask is None:
                continue
            self.data[i] = flip_polygons(mask, self.shape[1])
        return self

    def apply_resize(self, size=None, scale=None):
        """Apply resize transform."""
        if size is None:
            if not isinstance(scale, (tuple, list)):
                scale = (scale, scale)
            self.shape[0] = int(self.shape[0] * scale[0] + .5)
            self.shape[1] = int(self.shape[1] * scale[1] + .5)
        else:
            if not isinstance(size, (tuple, list)):
                size = (size, size)
            scale = (size[0] * 1. / self.shape[0],
                     size[1] * 1. / self.shape[1])
            self.shape = list(size)
        for mask in self.data:
            if mask is None:
                continue
            for p in mask:
                p[0::2] *= scale[1]
                p[1::2] *= scale[0]
        return self

    def apply_crop(self, crop_box):
        """Apply crop transform."""
        self.shape = [crop_box[3] - crop_box[1],
                      crop_box[2] - crop_box[0]]
        for i, mask in enumerate(self.data):
            if mask is None:
                continue
            self.data[i] = crop_polygons(mask, crop_box)

    def crop_and_resize(self, boxes, mask_size):
        """Return the resized ROI masks."""
        return [mask_from(self.data[i], mask_size, boxes[i])
                for i in range(len(self.data))]

    def get_boxes(self):
        """Return the bounding boxes of masks."""
        boxes = np.zeros((len(self.data), 4), 'float32')
        for i, mask in enumerate(self.data):
            if len(mask) == 0:
                continue
            xymin = np.array([float('inf'), float('inf')], 'float32')
            xymax = np.zeros((2,), 'float32')
            for p in mask:
                coords = p.reshape((-1, 2)).astype('float32')
                xymin = np.minimum(xymin, coords.min(0))
                xymax = np.maximum(xymax, coords.max(0))
            boxes[i, :2], boxes[i, 2:] = xymin, xymax
        return boxes

    def append(self, mask):
        """Append a mask."""
        assert isinstance(mask, list)
        self.data.append(mask)
        return self

    def extend(self, masks):
        """Append a set of masks."""
        for mask in masks:
            self.append(mask)
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.new_masks(self.data[item])
        elif isinstance(item, np.ndarray):
            return self.new_masks([self.data[i] for i in item.tolist()])
        return self.new_masks([self.data[item]])

    def __iadd__(self, masks):
        if isinstance(masks, PolygonMasks):
            self.data += masks.data
            return self
        return self.extend(masks)

    def __len__(self):
        return len(self.data)
