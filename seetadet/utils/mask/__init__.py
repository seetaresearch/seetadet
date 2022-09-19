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
"""Mask utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.utils.mask.helper import encode_masks
from seetadet.utils.mask.helper import mask_from
from seetadet.utils.mask.helper import mask_to_polygons
from seetadet.utils.mask.helper import paste_masks
from seetadet.utils.mask.metrics import mask_overlap
