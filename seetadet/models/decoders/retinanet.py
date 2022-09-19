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
"""RetinaNet decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import autograd
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.data.anchors.rpn import AnchorGenerator


class RetinaNetDecoder(nn.Module):
    """Decode predictions from retinanet."""

    def __init__(self):
        super(RetinaNetDecoder, self).__init__()
        self.anchor_generator = AnchorGenerator(
            strides=cfg.ANCHOR_GENERATOR.STRIDES,
            sizes=cfg.ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.ANCHOR_GENERATOR.ASPECT_RATIOS,
            scales_per_octave=3)
        self.pre_nms_topk = cfg.RETINANET.PRE_NMS_TOPK
        self.score_thresh = float(cfg.TEST.SCORE_THRESH)

    def forward(self, inputs):
        input_tags = ['cls_score', 'bbox_pred', 'im_info', 'grid_info']
        return autograd.Function.apply(
            'RetinaNetDecoder',
            inputs['cls_score'].device,
            inputs=[inputs[k] for k in input_tags],
            strides=self.anchor_generator.strides,
            ratios=self.anchor_generator.aspect_ratios[0],
            scales=self.anchor_generator.scales[0],
            pre_nms_topk=self.pre_nms_topk,
            score_thresh=self.score_thresh,
        )

    autograd.Function.register(
        'RetinaNetDecoder', lambda **kwargs: {
            'strides': kwargs.get('strides', []),
            'ratios': kwargs.get('ratios', []),
            'scales': kwargs.get('scales', []),
            'pre_nms_topk': kwargs.get('pre_nms_topk', 1000),
            'score_thresh': kwargs.get('score_thresh', 0.05),
            'check_device': False,
        })
