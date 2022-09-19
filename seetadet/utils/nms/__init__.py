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
"""Non-Maximum Suppression utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.utils.nms.helper import gpu_nms
from seetadet.utils.nms.helper import nms
from seetadet.utils.nms.helper import soft_nms
