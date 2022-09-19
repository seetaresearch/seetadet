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
"""Operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from seetadet.core.engine.utils import load_library as _load_library
_load_library(os.path.join(os.path.dirname(__file__), '_C'))
