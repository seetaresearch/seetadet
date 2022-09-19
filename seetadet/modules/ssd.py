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
"""SSD modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon.vm.torch as torch
import numpy as np

from seetadet.core.config import cfg
from seetadet.modules.build import InferenceModule
from seetadet.utils.bbox import bbox_transform_inv
from seetadet.utils.bbox import clip_boxes
from seetadet.utils.blob import blob_vstack
from seetadet.utils.image import im_rescale
from seetadet.utils.nms import nms


@InferenceModule.register('ssd')
class SSDInference(InferenceModule):
    """SSD inference module."""

    def __init__(self, model):
        super(SSDInference, self).__init__(model)
        self.forward_model = self.trace(
            'forward_eval', lambda self, img:
            self.forward({'img': img}))

    @torch.no_grad()
    def get_results(self, imgs):
        """Return the inference results."""
        results = self.forward_bbox(imgs)
        im_boxes = []
        for scores, boxes in results:
            with self.timers['misc'].tic_and_toc():
                cls_boxes = get_cls_results(scores, boxes)
            im_boxes.append(cls_boxes)
        return [{'boxes': boxes} for boxes in im_boxes]

    @torch.no_grad()
    def forward_data(self, imgs):
        """Return the inference data."""
        im_batch, im_scales = [], []
        for img in imgs:
            scaled_imgs, scales = im_rescale(
                img, scales=cfg.TEST.SCALES, keep_ratio=False)
            im_batch += scaled_imgs
            im_scales += scales
        im_batch = blob_vstack(im_batch, fill_value=cfg.MODEL.PIXEL_MEAN)
        return im_batch, im_scales

    @torch.no_grad()
    def forward_bbox(self, imgs):
        """Run bbox inference."""
        im_batch, im_scales = self.forward_data(imgs)
        self.timers['im_detect'].tic()
        inputs = {'img': torch.from_numpy(im_batch)}
        outputs = self.forward_model(inputs['img'])
        outputs = dict((k, outputs[k].numpy()) for k in outputs.keys())
        anchors = self.model.bbox_head.targets.generator.grid_anchors
        ims_per_batch, num_scales = len(imgs), len(cfg.TEST.SCALES)
        results = [([], []) for _ in range(ims_per_batch)]
        for i in range(ims_per_batch * num_scales):
            index = i // num_scales
            boxes = bbox_transform_inv(
                anchors, outputs['bbox_pred'][i],
                weights=cfg.SSD.BBOX_REG_WEIGHTS)
            boxes[:, 0::2] /= im_scales[i][1]
            boxes[:, 1::2] /= im_scales[i][0]
            boxes = clip_boxes(boxes, imgs[index].shape)
            results[index][0].append(outputs['cls_score'][i])
            results[index][1].append(boxes)
        results = [[np.vstack(x) for x in y] for y in results]
        self.timers['im_detect'].toc(n=ims_per_batch)
        return results


def get_cls_results(all_scores, all_boxes):
    """Return the categorical results."""
    cls_boxes = [[]]
    for j in range(1, len(cfg.MODEL.CLASSES)):
        inds = np.where(all_scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores, boxes = all_scores[inds, j], all_boxes[inds]
        inds = np.argsort(-scores)[:cfg.SSD.PRE_NMS_TOPK]
        scores, boxes = scores[inds], boxes[inds]
        dets = np.hstack((boxes, scores[:, np.newaxis]))
        dets = dets.astype('float32', copy=False)
        keep = nms(dets, cfg.TEST.NMS_THRESH)
        cls_boxes.append(dets[keep, :])
    return cls_boxes
