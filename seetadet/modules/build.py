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
"""Build for modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import types

import codewithgpu
from dragon.vm import torch

from seetadet.core.config import cfg
from seetadet.core.registry import Registry
from seetadet.utils.profiler import Timer

INFERENCE_MODULES = Registry('inference_modules')


def build_inference(model):
    """Build the inference module."""
    return INFERENCE_MODULES.get(cfg.MODEL.TYPE)(model)


class InferenceModule(codewithgpu.InferenceModule):
    """Inference module."""

    def __init__(self, model):
        super(InferenceModule, self).__init__(model)
        self.timers = collections.defaultdict(Timer)

    def get_time_diffs(self):
        """Return the time differences."""
        return dict((k, v.average_time)
                    for k, v in self.timers.items())

    def trace(self, name, func, example_inputs=None):
        """Trace the function and bound to model."""
        if not hasattr(self.model, name):
            setattr(self.model, name, torch.jit.trace(
                    func=types.MethodType(func, self.model),
                    example_inputs=example_inputs))
        return getattr(self.model, name)

    @staticmethod
    def register(model_type, **kwargs):
        """Register a inference module."""
        def decorated(func):
            return INFERENCE_MODULES.register(model_type, func, **kwargs)
        return decorated
