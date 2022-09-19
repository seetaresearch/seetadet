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
"""Detectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.models.detectors.detector import Detector
from seetadet.models.detectors.rcnn import CascadeRCNN
from seetadet.models.detectors.rcnn import FasterRCNN
from seetadet.models.detectors.rcnn import MaskRCNN
from seetadet.models.detectors.retinanet import RetinaNet
from seetadet.models.detectors.ssd import SSD
