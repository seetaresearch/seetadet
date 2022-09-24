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
"""RCNN modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torch
import numpy as np

from seetadet.core.config import cfg
from seetadet.modules.build import InferenceModule
from seetadet.utils.bbox import bbox_transform_inv
from seetadet.utils.bbox import clip_boxes
from seetadet.utils.bbox import distribute_boxes
from seetadet.utils.bbox import filter_empty_boxes
from seetadet.utils.blob import blob_vstack
from seetadet.utils.image import im_rescale
from seetadet.utils.nms import nms


@InferenceModule.register(['faster_rcnn', 'mask_rcnn', 'cascade_rcnn'])
class RCNNInference(InferenceModule):
    """RCNN inference module."""

    def __init__(self, model):
        super(RCNNInference, self).__init__(model)
        self.forward_model = self.trace(
            'forward_eval', lambda self, img, im_info, grid_info:
            self.forward({'img': img, 'im_info': im_info,
                          'grid_info': grid_info}))

    @torch.no_grad()
    def get_results(self, imgs):
        """Return the inference results."""
        img_boxes, proposals = self.forward_bbox(imgs)
        if getattr(self.model, 'mask_head', None) is None:
            return [{'boxes': boxes} for boxes in img_boxes]
        proposals = np.concatenate(sum(proposals, []))
        mask_pred = self.forward_mask(proposals)
        ims_per_batch, num_scales = len(imgs), len(cfg.TEST.SCALES)
        img_masks = [[] for _ in range(ims_per_batch)]
        batch_inds = proposals[:, :1].astype('int32')
        for i in range(ims_per_batch * num_scales):
            index = i // num_scales
            inds = np.where(batch_inds == i)[0]
            masks, labels = mask_pred[inds], proposals[inds, 5]
            num_classes = len(img_boxes[index])
            for _ in range(num_classes - len(img_masks[index])):
                img_masks[index].append([])
            for j in range(1, num_classes):
                img_masks[index][j].append(masks[np.where(labels == (j - 1))[0]])
                if (i + 1) % num_scales == 0:
                    v = img_masks[index][j]
                    img_masks[index][j] = np.vstack(v) if len(v) > 1 else v[0]
        return [{'boxes': boxes, 'masks': masks}
                for boxes, masks in zip(img_boxes, img_masks)]

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
        cls_score, bbox_pred = self.forward_cascade(outputs, im_info)
        ims_per_batch, num_scales = len(imgs), len(cfg.TEST.SCALES)
        results = [([], [], []) for _ in range(ims_per_batch)]
        batch_inds = outputs['rois'][:, :1].astype('int32')
        for i in range(ims_per_batch * num_scales):
            index = i // num_scales
            inds = np.where(batch_inds == i)[0]
            boxes = bbox_pred[inds] / im_info[i, 2]
            boxes = clip_boxes(boxes, imgs[index].shape)
            results[index][0].append(cls_score[inds])
            results[index][1].append(boxes)
            results[index][2].append(batch_inds[inds])
        results = [[np.vstack(x) for x in y] for y in results]
        self.timers['im_detect'].toc(n=ims_per_batch)
        img_boxes, img_proposals = [], []
        for scores, boxes, batch_inds in results:
            with self.timers['misc'].tic_and_toc():
                cls_boxes, cls_proposals = get_cls_results(
                    scores, boxes, batch_inds, im_info)
            img_boxes.append(cls_boxes)
            img_proposals.append(cls_proposals)
        return img_boxes, img_proposals

    @torch.no_grad()
    def forward_mask(self, proposals):
        """Run mask inference."""
        lvl_min, lvl_max = cfg.FAST_RCNN.MIN_LEVEL, cfg.FAST_RCNN.MAX_LEVEL
        lvls = distribute_boxes(proposals[:, 1:5], lvl_min, lvl_max)
        pool_inds = [np.where(lvls == (i + lvl_min))[0]
                     for i in range(lvl_max - lvl_min + 1)]
        restore_inds = np.concatenate(pool_inds).argsort()
        rois, labels = [], []
        for inds in pool_inds:
            rois.append(proposals[inds, :5] if len(inds) > 0 else
                        np.array([[-1, 0, 0, 1, 1]], 'float32'))
            labels.append(proposals[inds, 5].astype('int64')
                          if len(inds) > 0 else np.array([-1], 'int64'))
        self.timers['im_detect_mask'].tic()
        inputs = {'features': self.model.outputs['features'],
                  'rois': [self.model.to_tensor(x) for x in rois]}
        mask_pred = self.model.mask_head(inputs)['mask_pred']
        num_rois, num_classes = mask_pred.shape[:2]
        labels = np.concatenate(labels)
        fg_inds = np.where(labels >= 0)[0]
        strides = np.arange(num_rois) * num_classes
        mask_inds = self.model.to_tensor(strides[fg_inds] + labels[fg_inds])
        mask_pred = mask_pred.flatten_(0, 1)[mask_inds].numpy()
        mask_pred = mask_pred[restore_inds].copy()
        self.timers['im_detect_mask'].toc()
        return mask_pred

    @torch.no_grad()
    def forward_cascade(self, outputs, im_info):
        """Run cascade inference."""
        if not hasattr(self.model, 'bbox_heads'):
            bbox_pred = bbox_transform_inv(
                outputs['rois'][:, 1:5], outputs['bbox_pred'],
                weights=cfg.FAST_RCNN.BBOX_REG_WEIGHTS)
            return outputs['cls_score'], bbox_pred
        num_stages = len(self.model.bbox_heads)
        batch_inds = outputs['rois'][:, :1]
        ymax, xmax = np.split(im_info[batch_inds.flatten().astype('int32'), :2], 2, 1)
        lvl_min, lvl_max = cfg.FAST_RCNN.MIN_LEVEL, cfg.FAST_RCNN.MAX_LEVEL
        inputs = {'features': self.model.outputs['features']}
        cls_score = outputs['cls_score'].copy()
        valid_inds = restore_inds = bbox_pred = None
        for i in range(num_stages):
            if i > 0:
                outputs = self.model.bbox_heads[i](inputs)
                outputs = dict((k, outputs[k].numpy()) for k in outputs.keys())
                valid_inds = np.where(outputs['rois'][:, 0] > -2)[0]
                cls_score += outputs['cls_score'][valid_inds][restore_inds]
            bbox_pred = bbox_transform_inv(
                outputs['rois'][:, 1:5], outputs['bbox_pred'],
                weights=self.model.bbox_reg_weights[i])
            bbox_pred = bbox_pred[valid_inds][restore_inds] if i > 0 else bbox_pred
            if i < num_stages - 1:
                for k, v in zip([(0, 2), (1, 3)], [xmax, ymax]):
                    bbox_pred[:, k] = np.maximum(np.minimum(bbox_pred[:, k], v), 0)
                proposals = np.hstack((batch_inds, bbox_pred))
                lvls = distribute_boxes(bbox_pred, lvl_min, lvl_max)
                pool_inds = [np.where(lvls == (i + lvl_min))[0]
                             for i in range(lvl_max - lvl_min + 1)]
                restore_inds = np.concatenate(pool_inds).argsort()
                rois = [proposals[inds] if len(inds) > 0 else
                        np.array([[-2, 0, 0, 1, 1]], 'float32') for inds in pool_inds]
                inputs['rois'] = [self.model.to_tensor(x) for x in rois]
        cls_score *= 1.0 / num_stages
        return cls_score, bbox_pred


def get_cls_results(all_scores, all_boxes, batch_inds, im_info):
    """Return the categorical results."""
    empty_boxes = np.zeros((0, 5), 'float32')
    empty_proposals = np.zeros((0, 6), 'float32')
    cls_boxes, cls_proposals = [[]], []
    for j in range(1, len(cfg.MODEL.CLASSES)):
        inds = np.where(all_scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores = all_scores[inds, j]
        if cfg.FAST_RCNN.BBOX_REG_CLS_AGNOSTIC:
            boxes = all_boxes[inds]
        else:
            boxes = all_boxes[inds, (j - 1) * 4:j * 4]
        keep = filter_empty_boxes(boxes)
        if len(keep) == 0:
            cls_boxes.append(empty_boxes)
            cls_proposals.append(empty_proposals)
            continue
        scores, boxes = scores[keep], boxes[keep]
        dets = np.hstack((boxes, scores[:, np.newaxis]))
        dets = dets.astype('float32', copy=False)
        keep = nms(dets, cfg.TEST.NMS_THRESH)
        batch_inds_keep = batch_inds[inds][keep]
        cls_boxes.append(dets[keep, :])
        cls_proposals.append(np.hstack((
            batch_inds_keep,
            cls_boxes[-1][:, :4] * im_info[batch_inds_keep, 2],
            np.ones((len(keep), 1)) * (j - 1))).astype('float32'))
    return cls_boxes, cls_proposals
