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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr

from seetadet.core.config import cfg
from seetadet.data.structures import PolygonMasks
from seetadet.utils.bbox import bbox_overlaps
from seetadet.utils.bbox import clip_boxes
from seetadet.utils.bbox import flip_boxes
from seetadet.utils.image import im_resize
from seetadet.utils.image import color_jitter


class Transform(object):
    """Base transform type."""

    def init_params(self, params=None):
        for k, v in (params or {}).items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)

    def filter_outputs(self, *outputs):
        outputs = [x for x in outputs if x is not None]
        return outputs if len(outputs) > 1 else outputs[0]


class ParseBoxes(Transform):
    """Parse the ground-truth boxes."""

    def __init__(self):
        super(ParseBoxes, self).__init__()
        self.classes = cfg.MODEL.CLASSES
        self.num_classes = len(self.classes)
        self.class_indices = dict(zip(self.classes, range(self.num_classes)))
        self.use_diff = cfg.TRAIN.USE_DIFF

    def __call__(self, datum):
        height, width = datum.height, datum.width
        objects = list(filter(lambda obj: self.use_diff or
                              not obj.get('difficult', 0), datum.objects))
        boxes = np.empty((len(objects), 5), 'float32')
        for i, obj in enumerate(objects):
            boxes[i, :] = [max(0, obj['bbox'][0]),
                           max(0, obj['bbox'][1]),
                           min(obj['bbox'][2], width),
                           min(obj['bbox'][3], height),
                           self.class_indices[obj['name']]]
        return boxes


class ParseSegms(Transform):
    """Parse the ground-truth segmentations."""

    def __init__(self):
        super(ParseSegms, self).__init__()
        self.use_diff = cfg.TRAIN.USE_DIFF

    def __call__(self, datum):
        masks = PolygonMasks((datum.height, datum.width))
        objects = filter(lambda obj: self.use_diff or
                         not obj.get('difficult', 0), datum.objects)
        masks += [obj.get('polygons', None) for obj in objects]
        return masks


class RandomFlip(Transform):
    """Flip the image randomly."""

    def __init__(self, prob=0.5):
        super(RandomFlip, self).__init__()
        self.prob = prob
        self.is_flipped = False

    def __call__(self, img, boxes=None, segms=None):
        self.is_flipped = npr.rand() < self.prob
        img = img[:, ::-1] if self.is_flipped else img
        if self.is_flipped and boxes is not None:
            boxes = flip_boxes(boxes, img.shape[1])
        if self.is_flipped and segms is not None:
            segms = segms.apply_flip()
        return self.filter_outputs(img, boxes, segms)


class ResizeWarp(Transform):
    """Resize the image to a square size."""

    def __init__(self, size):
        super(ResizeWarp, self).__init__()
        self.size = size
        self.im_scale = (1.0, 1.0)

    def __call__(self, img, boxes=None):
        self.im_scale = (float(self.size) / float(img.shape[0]),
                         float(self.size) / float(img.shape[1]))
        img = im_resize(img, size=self.size)
        if boxes is not None:
            boxes[:, (0, 2)] = boxes[:, (0, 2)] * self.im_scale[1]
            boxes[:, (1, 3)] = boxes[:, (1, 3)] * self.im_scale[0]
        return self.filter_outputs(img, boxes)


class RandomResize(Transform):
    """Resize the image randomly."""

    def __init__(self, scales=(640,), scales_range=(1.0, 1.0), max_size=1066):
        super(RandomResize, self).__init__()
        self.scales = scales
        self.scales_range = scales_range
        self.max_size = max_size
        self.im_scale = 1.0
        self.scale_jitter = 1.0

    def __call__(self, img, boxes=None, segms=None):
        im_shape = img.shape
        target_size = npr.choice(self.scales)
        # Scale along the shortest side.
        max_size = max(self.max_size, target_size)
        im_size_min = np.min(im_shape[:2])
        im_size_max = np.max(im_shape[:2])
        self.im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than *MAX_SIZE*.
        if np.round(self.im_scale * im_size_max) > max_size:
            self.im_scale = float(max_size) / float(im_size_max)
        # Apply random scaling to get a range of dynamic scales.
        self.scale_jitter = npr.uniform(*self.scales_range)
        self.im_scale *= self.scale_jitter
        img = im_resize(img, scale=self.im_scale)
        if boxes is not None:
            boxes[:, :4] *= self.im_scale
        if segms is not None:
            segms.apply_resize(scale=self.im_scale)
        return self.filter_outputs(img, boxes, segms)


class RandomPaste(Transform):
    """Copy image into a larger canvas randomly."""

    def __init__(self, prob=0.5):
        self.ratio = 1. / cfg.TRAIN.SCALES_RANGE[0]
        self.prob = prob if self.ratio > 1 else 0
        self.pixel_mean = cfg.MODEL.PIXEL_MEAN

    def __call__(self, img, boxes):
        if npr.rand() > self.prob:
            return img, boxes
        im_shape = list(img.shape)
        h, w = im_shape[:2]
        ratio = npr.uniform(1., self.ratio)
        out_h, out_w = int(h * ratio), int(w * ratio)
        y1 = int(np.floor(npr.uniform(0., out_h - h)))
        x1 = int(np.floor(npr.uniform(0., out_w - w)))
        im_shape[:2] = (out_h, out_w)
        out_img = np.empty(im_shape, img.dtype)
        out_img[:] = self.pixel_mean
        out_img[y1:y1 + h, x1:x1 + w, :] = img
        out_boxes = boxes.astype(boxes.dtype, copy=True)
        out_boxes[:, (0, 2)] = (boxes[:, (0, 2)] * w + x1) / out_w
        out_boxes[:, (1, 3)] = (boxes[:, (1, 3)] * h + y1) / out_h
        return out_img, out_boxes


class RandomCrop(Transform):
    """Crop the image randomly."""

    def __init__(self, crop_size=512):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.pixel_mean = cfg.MODEL.PIXEL_MEAN

    def __call__(self, img, boxes=None, segms=None):
        if self.crop_size <= 0:
            return self.filter_outputs(img, boxes, segms)
        im_shape = list(img.shape)
        h, w = im_shape[:2]
        out_h, out_w = (self.crop_size,) * 2
        y = npr.randint(max(h - out_h, 0) + 1)
        x = npr.randint(max(w - out_w, 0) + 1)
        im_shape[:2] = (out_h, out_w)
        out_img = np.empty(im_shape, img.dtype)
        out_img[:] = self.pixel_mean
        out_img[:h, :w] = img[y:y + out_h, x:x + out_w]
        img = out_img
        if boxes is not None:
            boxes[:, (0, 2)] -= x
            boxes[:, (1, 3)] -= y
        if segms is not None:
            segms.apply_crop((x, y, x + out_w, y + out_h))
        return self.filter_outputs(img, boxes, segms)


class ColorJitter(Transform):
    """Distort the brightness, contrast and color of image."""

    def __init__(self, prob=0.5):
        super(ColorJitter, self).__init__()
        self.prob = prob
        self.brightness_range = (0.875, 1.125)
        self.contrast_range = (0.5, 1.5)
        self.saturation_range = (0.5, 1.5)

    def __call__(self, img):
        brightness = contrast = saturation = None
        if npr.rand() < self.prob:
            brightness = self.brightness_range
        if npr.rand() < self.prob:
            contrast = self.contrast_range
        if npr.rand() < self.prob:
            saturation = self.saturation_range
        return color_jitter(img, brightness=brightness,
                            contrast=contrast, saturation=saturation)


class RandomBBoxCrop(Transform):
    """Crop image by sampling a region restricted by bounding boxes."""

    def __init__(self, scales_range=(0.3, 1.0), aspect_ratios_range=(0.5, 2.0),
                 overlaps=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9)):
        self.samplers = [{}]
        for ov in overlaps:
            self.samplers.append({
                'scales_range': scales_range,
                'aspect_ratios_range': aspect_ratios_range,
                'overlaps_range': (ov, 1.0), 'max_trials': 10})

    @staticmethod
    def generate_sample(param):
        scales_range = param.get('scales_range', (1.0, 1.0))
        aspect_ratios_range = param.get('aspect_ratios_range', (1.0, 1.0))
        scale = npr.uniform(scales_range[0], scales_range[1])
        min_aspect_ratio = max(aspect_ratios_range[0], scale**2)
        max_aspect_ratio = min(aspect_ratios_range[1], 1. / (scale**2))
        aspect_ratio = npr.uniform(min_aspect_ratio, max_aspect_ratio)
        bbox_w = scale * (aspect_ratio ** 0.5)
        bbox_h = scale / (aspect_ratio ** 0.5)
        w_off = npr.uniform(0., 1. - bbox_w)
        h_off = npr.uniform(0., 1. - bbox_h)
        return np.array([w_off, h_off, w_off + bbox_w, h_off + bbox_h])

    @staticmethod
    def check_center(sample_box, boxes):
        x_ctr = (boxes[:, 2] + boxes[:, 0]) / 2.0
        y_ctr = (boxes[:, 3] + boxes[:, 1]) / 2.0
        keep = np.where((x_ctr >= sample_box[0]) & (x_ctr <= sample_box[2]) &
                        (y_ctr >= sample_box[1]) & (y_ctr <= sample_box[3]))[0]
        return len(keep) > 0

    @staticmethod
    def check_overlap(sample_box, boxes, param):
        ov_range = param.get('overlaps_range', (0.0, 1.0))
        if ov_range[0] == 0.0 and ov_range[1] == 1.0:
            return True
        ovmax = bbox_overlaps(sample_box[None, :], boxes[:, :4]).max()
        if ovmax < ov_range[0] or ovmax > ov_range[1]:
            return False
        return True

    def generate_samples(self, boxes):
        crop_boxes = []
        for sampler in self.samplers:
            for _ in range(sampler.get('max_trials', 1)):
                crop_box = self.generate_sample(sampler)
                if not self.check_overlap(crop_box, boxes, sampler):
                    continue
                if not self.check_center(crop_box, boxes):
                    continue
                crop_boxes.append(crop_box)
                break
        return crop_boxes

    @classmethod
    def crop(cls, img, crop_box, boxes=None):
        h, w = img.shape[:2]
        w_offset = int(crop_box[0] * w)
        h_offset = int(crop_box[1] * h)
        crop_w = int((crop_box[2] - crop_box[0]) * w)
        crop_h = int((crop_box[3] - crop_box[1]) * h)
        img = img[h_offset:h_offset + crop_h, w_offset:w_offset + crop_w]
        if boxes is not None:
            x_ctr = (boxes[:, 2] + boxes[:, 0]) / 2.0
            y_ctr = (boxes[:, 3] + boxes[:, 1]) / 2.0
            keep = np.where((x_ctr >= crop_box[0]) & (x_ctr <= crop_box[2]) &
                            (y_ctr >= crop_box[1]) & (y_ctr <= crop_box[3]))[0]
            boxes = boxes[keep]
            boxes[:, (0, 2)] = boxes[:, (0, 2)] * w - w_offset
            boxes[:, (1, 3)] = boxes[:, (1, 3)] * h - h_offset
            boxes = clip_boxes(boxes, (crop_h, crop_w))
            boxes[:, (0, 2)] /= crop_w
            boxes[:, (1, 3)] /= crop_h
        return img, boxes

    def __call__(self, img, boxes):
        crop_boxes = self.generate_samples(boxes)
        if len(crop_boxes) > 0:
            crop_box = crop_boxes[npr.randint(len(crop_boxes))]
            img, boxes = self.crop(img, crop_box, boxes)
        return img, boxes
