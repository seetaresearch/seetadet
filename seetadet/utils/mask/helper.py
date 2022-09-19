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
"""Helper functions for mask."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import cv2
import numpy as np
from pycocotools.mask import decode
from pycocotools.mask import encode
from pycocotools.mask import merge
from pycocotools.mask import frPyObjects

from seetadet.ops.normalization import to_tensor
from seetadet.ops.vision import PasteMask
from seetadet.utils.image import im_resize


def mask_from_buffer(buffer, size, box=None):
    """Return a binary mask from the buffer."""
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    rles = [{'counts': buffer, 'size': size}]
    mask = decode(rles)
    if mask.shape[2] != 1:
        raise ValueError('Mask contains {} instances. '
                         'Merge them before compressing.'
                         .format(mask.shape[2]))
    mask = mask[:, :, 0]
    if box is not None:
        box = np.round(box).astype('int64')
        mask = mask[box[1]:box[3], box[0]:box[2]]
    return mask


def mask_from_polygons(polygons, size, box=None):
    """Return a binary mask from the polygons."""
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    if box is not None:
        polygons = copy.deepcopy(polygons)
        w, h = box[2] - box[0], box[3] - box[1]
        ratio_h = size[0] / max(h, 0.1)
        ratio_w = size[1] / max(w, 0.1)
        for p in polygons:
            p[0::2] = p[0::2] - box[0]
            p[1::2] = p[1::2] - box[1]
        if ratio_h == ratio_w:
            for p in polygons:
                p *= ratio_h
        else:
            for p in polygons:
                p[0::2] *= ratio_w
                p[1::2] *= ratio_h
    rles = frPyObjects(polygons, size[0], size[1])
    return decode(merge(rles))


def mask_from_bitmap(bitmap, size, box=None):
    """Return a binary mask from the bitmap."""
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    if box is not None:
        box = np.round(box).astype('int64')
        bitmap = bitmap[box[1]:box[3], box[0]:box[2]]
    return im_resize(bitmap, size, mode='nearest')


def mask_from(segm, size, box=None):
    """Return a binary mask from the segmentation object."""
    if segm is None:
        return None
    elif isinstance(segm, list):
        return mask_from_polygons(segm, size, box)
    elif isinstance(segm, np.ndarray):
        return mask_from_bitmap(segm, size, box)
    elif isinstance(segm, bytes):
        return mask_from_buffer(segm, size, box)
    else:
        raise TypeError('Unknown segmentation type: ' + type(segm))


def mask_to_polygons(mask):
    """Convert a binary mask to a set of polygons."""
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:
        return []
    contours = res[-2]
    polygons = [x.flatten() for x in contours]
    polygons = [x + 0.5 for x in polygons if len(x) >= 6]
    return polygons


def encode_masks(masks):
    """Encode a set of masks to RLEs."""
    rles = encode(np.asfortranarray(masks))
    for rle in rles:
        rle['counts'] = rle['counts'].decode()
    return rles


def paste_masks(masks, boxes, img_size, threshold=0.5, channels_last=True):
    """Paste a set of masks on an image by resample."""
    masks, boxes = to_tensor(masks), to_tensor(boxes[:, :4])
    img_masks = PasteMask.apply(masks, boxes, img_size, threshold)
    img_masks = img_masks.numpy().copy()
    return img_masks.transpose((1, 2, 0)) if channels_last else img_masks


def paste_masks_old(masks, boxes, img_size, thresh=0.5):
    """Paste a set of masks on an image by resize."""
    def scale_boxes(boxes, scale_factor=1.):
        """Scale the boxes."""
        w = (boxes[:, 2] - boxes[:, 0]) * 0.5 * scale_factor
        h = (boxes[:, 3] - boxes[:, 1]) * 0.5 * scale_factor
        x_ctr = (boxes[:, 2] + boxes[:, 0]) * 0.5
        y_ctr = (boxes[:, 3] + boxes[:, 1]) * 0.5
        boxes_scaled = np.zeros(boxes.shape)
        boxes_scaled[:, 0], boxes_scaled[:, 1] = x_ctr - w, y_ctr - h
        boxes_scaled[:, 2], boxes_scaled[:, 3] = x_ctr + w, y_ctr + h
        return boxes_scaled
    num_boxes = boxes.shape[0]
    assert masks.shape[0] == num_boxes
    img_shape = list(img_size) + [num_boxes]
    output = np.zeros(img_shape, 'uint8')
    size = masks[0].shape[0]
    scale_factor = (size + 2.) / size
    boxes = scale_boxes(boxes, scale_factor).astype(np.int32)
    padded_mask = np.zeros((size + 2, size + 2), 'float32')
    for i in range(num_boxes):
        box, mask = boxes[i, :4], masks[i]
        padded_mask[1:-1, 1:-1] = mask[:, :]
        w = max(box[2] - box[0], 1)
        h = max(box[3] - box[1], 1)
        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > thresh, 'uint8')
        x1, y1 = max(box[0], 0), max(box[1], 0)
        x2, y2 = min(box[2], img_size[1]), min(box[3], img_size[0])
        mask = mask[y1 - box[1]:y2 - box[1], x1 - box[0]:x2 - box[0]]
        output[y1:y2, x1:x2, i] = mask
    return output
