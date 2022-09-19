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
"""Base detector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import build_backbone
from seetadet.ops.fusion import get_fusion
from seetadet.ops.normalization import ToTensor
from seetadet.utils import logging


class Detector(nn.Module):
    """Class to build and compute the detection pipelines."""

    def __init__(self):
        super(Detector, self).__init__()
        self.to_tensor = ToTensor()
        self.backbone, self.neck = build_backbone()
        self.backbone_dims = self.neck.out_dims

    def get_inputs(self, inputs):
        """Return the detection inputs.

        Parameters
        ----------
        inputs : dict, optional
            The optional inputs.

        """
        inputs['img'] = self.to_tensor(inputs['img'], normalize=True)
        return inputs

    def get_features(self, inputs):
        """Return the detection features.

        Parameters
        ----------
        inputs : dict
            The inputs.

        """
        return self.neck(self.backbone(inputs['img']))

    def get_outputs(self, inputs):
        """Return the detection outputs.

        Parameters
        ----------
        inputs : dict
            The inputs.

        """
        return inputs

    def forward(self, inputs):
        """Define the computation performed at every call.

        Parameters
        ----------
        inputs : dict
            The inputs.

        """
        return self.get_outputs(inputs)

    def load_weights(self, weights, strict=False):
        """Load the state dict of this detector.

        Parameters
        ----------
        weights : str
            The path of the weights file.

        """
        return self.load_state_dict(torch.load(weights), strict=strict)

    def optimize_for_inference(self):
        """Optimize the graph for the inference."""
        # Set precision.
        precision = cfg.MODEL.PRECISION.lower()
        self.half() if precision == 'float16' else self.float()
        logging.info('Set precision: ' + precision)
        # Fuse modules.
        fusion_memo, last_module = set(), None
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, 'optimize_for_inference'):
                module.optimize_for_inference()
                fusion_memo.add(module.__class__.__name__)
                continue
            key, fn = get_fusion(last_module, module)
            if fn is not None:
                fusion_memo.add(key)
                fn(last_module, module)
            last_module = module
        for key in fusion_memo:
            logging.info('Fuse modules: ' + key)
