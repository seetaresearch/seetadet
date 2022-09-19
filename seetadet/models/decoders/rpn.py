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
"""RPN decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import autograd
from dragon.vm.torch import nn

import numpy as np

from seetadet.core.config import cfg
from seetadet.data.anchors.rpn import AnchorGenerator
from seetadet.utils.bbox import bbox_transform_inv
from seetadet.utils.bbox import clip_boxes
from seetadet.utils.bbox import filter_empty_boxes
from seetadet.utils.nms import gpu_nms


class RPNDecoder(nn.Module):
    """Generate proposal regions from RPN."""

    def __init__(self):
        super(RPNDecoder, self).__init__()
        self.anchor_generator = AnchorGenerator(
            strides=cfg.ANCHOR_GENERATOR.STRIDES,
            sizes=cfg.ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.ANCHOR_GENERATOR.ASPECT_RATIOS)
        self.min_level = cfg.FAST_RCNN.MIN_LEVEL
        self.max_level = cfg.FAST_RCNN.MAX_LEVEL
        self.pre_nms_topk = {True: cfg.RPN.PRE_NMS_TOPK_TRAIN,
                             False: cfg.RPN.PRE_NMS_TOPK_TEST}
        self.post_nms_topk = {True: cfg.RPN.POST_NMS_TOPK_TRAIN,
                              False: cfg.RPN.POST_NMS_TOPK_TEST}
        self.nms_thresh = float(cfg.RPN.NMS_THRESH)

    def decode_proposals(self, scores, deltas, anchors, im_info):
        # Select top-K anchors.
        pre_nms_topk = self.pre_nms_topk[self.training]
        if pre_nms_topk <= 0 or pre_nms_topk >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            inds = np.argpartition(-scores.squeeze(), pre_nms_topk)[:pre_nms_topk]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        scores, deltas, anchors = scores[order], deltas[order], anchors[order]
        # Convert anchors into proposals.
        proposals = bbox_transform_inv(anchors, deltas)
        proposals = clip_boxes(proposals, im_info[:2])
        keep = filter_empty_boxes(proposals)
        if len(proposals) != len(keep):
            proposals, scores = proposals[keep], scores[keep]
        # Apply NMS.
        proposals = np.hstack((proposals, scores))
        keep = gpu_nms(proposals, self.nms_thresh)
        return proposals[keep, :].astype('float32', copy=False)

    def forward_train(self, inputs):
        shapes = [x[:2] for x in inputs['grid_info']]
        anchors = self.anchor_generator.get_anchors(shapes)
        cls_score = inputs['cls_score'].numpy()
        bbox_pred = inputs['bbox_pred'].permute(0, 2, 1).numpy()
        all_rois, batch_size = [], cls_score.shape[0]
        lvl_slices, lvl_start = [], 0
        post_nms_topk = self.post_nms_topk[self.training]
        for shape in shapes:
            num_anchors = self.anchor_generator.num_anchors([shape])
            lvl_slices.append(slice(lvl_start, lvl_start + num_anchors))
            lvl_start = lvl_start + num_anchors
        for batch_ind in range(batch_size):
            scores = cls_score[batch_ind].reshape((-1, 1))
            deltas = bbox_pred[batch_ind]
            im_info = inputs['im_info'][batch_ind]
            all_proposals = []
            for lvl_slice in lvl_slices:
                all_proposals.append(self.decode_proposals(
                    scores[lvl_slice], deltas[lvl_slice],
                    anchors[lvl_slice], im_info))
            proposals = np.concatenate(all_proposals)
            proposals, scores = proposals[:, :4], proposals[:, -1]
            if post_nms_topk > 0:
                keep = np.argsort(-scores)[:post_nms_topk]
                proposals = proposals[keep, :]
            batch_inds = np.full((proposals.shape[0], 1), batch_ind, 'float32')
            all_rois.append(np.hstack((batch_inds, proposals)))
        return np.concatenate(all_rois)

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        input_tags = ['cls_score', 'bbox_pred', 'im_info', 'grid_info']
        return autograd.Function.apply(
            'RPNDecoder',
            inputs['cls_score'].device,
            inputs=[inputs[k] for k in input_tags],
            outputs=[None] * (self.max_level - self.min_level + 1),
            strides=self.anchor_generator.strides,
            ratios=self.anchor_generator.aspect_ratios[0],
            scales=self.anchor_generator.scales[0],
            min_level=self.min_level,
            max_level=self.max_level,
            pre_nms_topk=self.pre_nms_topk[False],
            post_nms_topk=self.post_nms_topk[False],
            nms_thresh=self.nms_thresh,
        )

    autograd.Function.register(
        'RPNDecoder', lambda **kwargs: {
            'strides': kwargs.get('strides', []),
            'ratios': kwargs.get('ratios', []),
            'scales': kwargs.get('scales', []),
            'pre_nms_topk': kwargs.get('pre_nms_topk', 1000),
            'post_nms_topk': kwargs.get('post_nms_topk', 1000),
            'nms_thresh': kwargs.get('nms_thresh', 0.7),
            'min_level': kwargs.get('min_level', 2),
            'max_level': kwargs.get('max_level', 5),
            'check_device': False,
        })
