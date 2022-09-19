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
"""RetinaNet detector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.models.build import DETECTORS
from seetadet.models.decoders.retinanet import RetinaNetDecoder
from seetadet.models.dense_heads.retinanet import RetinaNetHead
from seetadet.models.detectors.detector import Detector


@DETECTORS.register('retinanet')
class RetinaNet(Detector):
    """RetinaNet detector."""

    def __init__(self):
        super(RetinaNet, self).__init__()
        self.bbox_head = RetinaNetHead(self.backbone_dims)
        self.bbox_decoder = RetinaNetDecoder()

    def get_outputs(self, inputs):
        """Compute detection outputs."""
        inputs = self.get_inputs(inputs)
        inputs['features'] = self.get_features(inputs)
        inputs['grid_info'] = inputs.pop(
            'grid_info', [x.shape[-2:] for x in inputs['features']])
        outputs = self.bbox_head(inputs)
        if not self.training:
            outputs['dets'] = self.bbox_decoder({
                'cls_score': outputs.pop('cls_score'),
                'bbox_pred': outputs.pop('bbox_pred'),
                'im_info': inputs['im_info'],
                'grid_info': inputs['grid_info']})
        return outputs
