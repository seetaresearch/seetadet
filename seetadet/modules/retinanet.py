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
"""RetinaNet modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon.vm.torch as torch
import numpy as np

from seetadet.core.config import cfg
from seetadet.modules.build import InferenceModule
from seetadet.utils.blob import blob_vstack
from seetadet.utils.image import im_rescale
from seetadet.utils.nms import nms


@InferenceModule.register('retinanet')
class RetinaNetInference(InferenceModule):
    """RetinaNet inference module."""

    def __init__(self, model):
        super(RetinaNetInference, self).__init__(model)
        self.forward_model = self.trace(
            'forward_eval', lambda self, img, im_info, grid_info:
            self.forward({'img': img, 'im_info': im_info,
                          'grid_info': grid_info}))

    @torch.no_grad()
    def get_results(self, imgs):
        """Return the inference results."""
        results = self.forward_bbox(imgs)
        img_boxes = []
        for dets in results:
            with self.timers['misc'].tic_and_toc():
                cls_boxes = get_cls_results(dets)
            img_boxes.append(cls_boxes)
        return [{'boxes': boxes} for boxes in img_boxes]

    @torch.no_grad()
    def forward_data(self, imgs):
        """Return the inference data."""
        im_batch, im_shapes, im_scales = [], [], []
        for img in imgs:
            scaled_imgs, scales = im_rescale(
                img, scales=cfg.TEST.SCALES, max_size=cfg.TEST.MAX_SIZE)
            im_batch += scaled_imgs
            im_scales += scales
            im_shapes += [x.shape[:2] for x in scaled_imgs]
        im_batch = blob_vstack(
            im_batch, fill_value=cfg.MODEL.PIXEL_MEAN,
            size=(cfg.TEST.CROP_SIZE,) * 2,
            align=(cfg.BACKBONE.COARSEST_STRIDE,) * 2)
        im_shapes = np.array(im_shapes)
        im_scales = np.array(im_scales).reshape((len(im_batch), -1))
        im_info = np.hstack([im_shapes, im_scales]).astype('float32')
        strides = [2 ** x for x in range(cfg.FPN.MIN_LEVEL, cfg.FPN.MAX_LEVEL + 1)]
        strides = np.array(strides)[:, None]
        grid_shapes = np.stack([im_batch.shape[1:3]] * len(strides))
        grid_shapes = (grid_shapes - 1) // strides + 1
        grid_info = grid_shapes.astype('int64')
        return im_batch, im_info, grid_info

    @torch.no_grad()
    def forward_bbox(self, imgs):
        """Run bbox inference."""
        im_batch, im_info, grid_info = self.forward_data(imgs)
        self.timers['im_detect'].tic()
        inputs = {'img': torch.from_numpy(im_batch),
                  'im_info': torch.from_numpy(im_info),
                  'grid_info': torch.from_numpy(grid_info)}
        outputs = self.forward_model(inputs['img'], inputs['im_info'],
                                     inputs['grid_info'])
        outputs = dict((k, outputs[k].numpy()) for k in outputs.keys())
        ims_per_batch, num_scales = len(imgs), len(cfg.TEST.SCALES)
        results = [[] for _ in range(ims_per_batch)]
        batch_inds = outputs['dets'][:, 0:1].astype('int32')
        for i in range(ims_per_batch * num_scales):
            index = i // num_scales
            inds = np.where(batch_inds == i)[0]
            results[index].append(outputs['dets'][inds, 1:])
        for index in range(ims_per_batch):
            try:
                results[index] = np.vstack(results[index])
            except ValueError:
                results[index] = results[index][0]
        self.timers['im_detect'].toc(n=ims_per_batch)
        return results


def get_cls_results(all_dets):
    """Return the categorical results."""
    empty_boxes = np.zeros((0, 5), 'float32')
    cls_boxes = [[]]
    labels = all_dets[:, 5].astype('int32')
    for j in range(1, len(cfg.MODEL.CLASSES)):
        inds = np.where(labels == j)[0]
        if len(inds) == 0:
            cls_boxes.append(empty_boxes)
            continue
        dets = all_dets[inds, :5].astype('float32')
        keep = nms(dets, cfg.TEST.NMS_THRESH)
        cls_boxes.append(dets[keep, :])
    return cls_boxes
