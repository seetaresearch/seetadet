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
"""SSD detector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.models.build import DETECTORS
from seetadet.models.dense_heads.ssd import SSDHead
from seetadet.models.detectors.detector import Detector


@DETECTORS.register('ssd')
class SSD(Detector):
    """SSD detector."""

    def __init__(self):
        super(SSD, self).__init__()
        self.bbox_head = SSDHead(self.backbone_dims)

    def get_outputs(self, inputs=None):
        """Compute detection outputs."""
        inputs = self.get_inputs(inputs)
        inputs['features'] = self.get_features(inputs)
        outputs = self.bbox_head(inputs)
        return outputs
