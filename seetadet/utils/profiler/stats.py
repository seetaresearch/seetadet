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
"""Trackable statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np


class SmoothedValue(object):
    """Track values and provide smoothed report."""

    def __init__(self, window_size=None):
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def mean(self):
        return np.mean(self.deque)

    def median(self):
        return np.median(self.deque)

    def average(self):
        return self.total / self.count


class ExponentialMovingAverage(object):
    """Track values and provide EMA report."""

    def __init__(self, decay=0.9):
        self.value = None
        self.decay = decay
        self.total = 0.0
        self.count = 0

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1. - self.decay) * value
        self.total += value
        self.count += 1
        return self.running_average()

    def average(self):
        return self.total / self.count

    def running_average(self):
        return self.value

    def __float__(self):
        return self.running_average()
