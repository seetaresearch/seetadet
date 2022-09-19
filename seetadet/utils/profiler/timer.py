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
"""Timing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import datetime
import time


class Timer(object):
    """Simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def add_diff(self, diff, n=1, average=True):
        self.total_time += diff
        self.calls += n
        self.average_time = self.total_time / self.calls
        return self.average_time if average else self.diff

    @contextlib.contextmanager
    def tic_and_toc(self, n=1, average=True):
        try:
            yield self.tic()
        finally:
            self.toc(n, average)

    def tic(self):
        self.start_time = time.time()
        return self

    def toc(self, n=1, average=True):
        self.diff = time.time() - self.start_time
        return self.add_diff(self.diff, n, average)


def get_progress(timer, step, max_steps):
    """Return the progress information."""
    eta_seconds = timer.average_time * (max_steps - step)
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    progress = (step + 1.) / max_steps
    return ('< PROGRESS: {:.2%} | SPEED: {:.3f}s / iter | ETA: {} >'
            .format(progress, timer.average_time, eta))
