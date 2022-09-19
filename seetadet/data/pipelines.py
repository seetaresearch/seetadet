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
"""Data loading pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import cv2
import numpy as np

from seetadet.core.config import cfg
from seetadet.data import transforms
from seetadet.data.build import LOADERS
from seetadet.data.build import build_anchor_sampler
from seetadet.data.datasets import AnnotatedDatum
from seetadet.data.loader import DataLoader
from seetadet.utils.bbox import clip_boxes
from seetadet.utils.bbox import filter_empty_boxes


class WorkerBase(multiprocessing.Process):
    """Base class of data worker."""

    def __init__(self):
        super(WorkerBase, self).__init__(daemon=True)
        self.seed = cfg.RNG_SEED
        self.reader_queue = None
        self.worker_queue = None

    def get_outputs(self, inputs):
        """Return the processed outputs."""
        return inputs

    def run(self):
        """Main prefetch loop."""
        # Disable the opencv threading.
        cv2.setNumThreads(1)
        # Fix the process-local random seed.
        np.random.seed(self.seed)
        inputs = []
        while True:
            # Use cached buffer for next 4 inputs.
            while len(inputs) < 4:
                inputs.append(self.reader_queue.get())
            outputs = self.get_outputs(inputs)
            self.worker_queue.put(outputs)


class DetTrainWorker(WorkerBase):
    """Generic train pipeline for detection."""

    def __init__(self, **kwargs):
        super(DetTrainWorker, self).__init__()
        self.parse_boxes = transforms.ParseBoxes()
        self.resize = transforms.RandomResize(
            scales=cfg.TRAIN.SCALES,
            scales_range=cfg.TRAIN.SCALES_RANGE,
            max_size=cfg.TRAIN.MAX_SIZE)
        self.flip = transforms.RandomFlip()
        self.crop = transforms.RandomCrop(cfg.TRAIN.CROP_SIZE)
        self.distort = transforms.ColorJitter(cfg.TRAIN.COLOR_JITTER)
        self.anchor_sampler = build_anchor_sampler()

    def get_outputs(self, inputs):
        datum = AnnotatedDatum(inputs.pop(0))
        img, boxes = datum.img, self.parse_boxes(datum)
        img, boxes = self.resize(img, boxes)
        img, boxes = self.flip(img, boxes)
        img, boxes = self.crop(img, boxes)
        boxes = clip_boxes(boxes, img.shape)
        boxes = boxes[filter_empty_boxes(boxes)]
        if len(boxes) == 0:
            return None
        img = self.distort(img)
        im_scale = self.resize.im_scale
        aspect_ratio = float(img.shape[0]) / float(img.shape[1])
        outputs = {'img': [img],
                   'gt_boxes': [boxes],
                   'im_info': [img.shape[:2] + (im_scale,)],
                   'aspect_ratio': [aspect_ratio]}
        if self.anchor_sampler is not None:
            data = self.anchor_sampler.sample(boxes)
            for k, v in data.items():
                outputs[k] = [v]
        return outputs


class MaskTrainWorker(WorkerBase):
    """Generic train pipeline for instance segmentation."""

    def __init__(self, **kwargs):
        super(MaskTrainWorker, self).__init__()
        self.parse_boxes = transforms.ParseBoxes()
        self.parse_segms = transforms.ParseSegms()
        self.resize = transforms.RandomResize(
            scales=cfg.TRAIN.SCALES,
            scales_range=cfg.TRAIN.SCALES_RANGE,
            max_size=cfg.TRAIN.MAX_SIZE)
        self.flip = transforms.RandomFlip()
        self.crop = transforms.RandomCrop(cfg.TRAIN.CROP_SIZE)
        self.distort = transforms.ColorJitter(cfg.TRAIN.COLOR_JITTER)
        self.recompute_boxes = cfg.TRAIN.CROP_SIZE > 0
        self.anchor_sampler = build_anchor_sampler()

    def get_outputs(self, inputs):
        datum = AnnotatedDatum(inputs.pop(0))
        img, boxes = datum.img, self.parse_boxes(datum)
        segms = self.parse_segms(datum)
        img, boxes, segms = self.resize(img, boxes, segms)
        img, boxes, segms = self.flip(img, boxes, segms)
        img, boxes, segms = self.crop(img, boxes, segms)
        if self.recompute_boxes:
            boxes[:, :4] = segms.get_boxes()
        else:
            boxes = clip_boxes(boxes, img.shape)
        keep = filter_empty_boxes(boxes)
        boxes, segms = boxes[keep], segms[keep]
        if len(boxes) == 0:
            return None
        img = self.distort(img)
        im_scale = self.resize.im_scale
        aspect_ratio = float(img.shape[0]) / float(img.shape[1])
        outputs = {'img': [img],
                   'gt_boxes': [boxes],
                   'gt_segms': [segms],
                   'im_info': [img.shape[:2] + (im_scale,)],
                   'scale_jitter': [self.resize.scale_jitter],
                   'aspect_ratio': [aspect_ratio]}
        if self.anchor_sampler is not None:
            data = self.anchor_sampler.sample(boxes)
            for k, v in data.items():
                outputs[k] = [v]
        return outputs


class SSDTrainWorker(WorkerBase):
    """Generic train pipeline for SSD detection."""

    def __init__(self, **kwargs):
        super(SSDTrainWorker, self).__init__()
        self.parse_boxes = transforms.ParseBoxes()
        self.paste = transforms.RandomPaste()
        self.crop = transforms.RandomBBoxCrop()
        self.resize = transforms.ResizeWarp(cfg.TRAIN.SCALES[0])
        self.flip = transforms.RandomFlip()
        self.distort = transforms.ColorJitter(cfg.TRAIN.COLOR_JITTER)
        self.anchor_sampler = build_anchor_sampler()

    def get_outputs(self, inputs):
        datum = AnnotatedDatum(inputs.pop(0))
        img, boxes = datum.img, self.parse_boxes(datum)
        boxes /= [(img.shape[1], img.shape[0]) * 2 + (1,)]
        img, boxes = self.paste(img, boxes)
        img, boxes = self.crop(img, boxes)
        if len(boxes) == 0:
            return None
        img = self.resize(img)
        boxes[:, :4] *= img.shape[0]
        img, boxes = self.flip(img, boxes)
        img = self.distort(img)
        outputs = {'img': [img],
                   'gt_boxes': [boxes],
                   'im_info': [img.shape[:2]]}
        if self.anchor_sampler is not None:
            data = self.anchor_sampler.sample(boxes)
            for k, v in data.items():
                outputs[k] = [v]
        return outputs


class DetTestWorker(WorkerBase):
    """Generic test pipeline for detection."""

    def __init__(self, **kwargs):
        super(DetTestWorker, self).__init__()

    def get_outputs(self, inputs):
        datum = AnnotatedDatum(inputs.pop(0))
        img, objects = datum.img, datum.objects
        outputs = {'img': [img], 'objects': [objects],
                   'img_meta': [{'id': datum.id,
                                 'height': datum.height,
                                 'width': datum.width}]}
        return outputs


LOADERS.register('det_train', DataLoader, worker=DetTrainWorker)
LOADERS.register('mask_train', DataLoader, worker=MaskTrainWorker)
LOADERS.register('ssd_train', DataLoader, worker=SSDTrainWorker)
LOADERS.register('det_test', DataLoader, worker=DetTestWorker)
