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
"""Build for models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.core.registry import Registry
from seetadet.core.engine.utils import get_device

BACKBONES = Registry('backbones')
NECKS = Registry('necks')
DETECTORS = Registry('detectors')


def build_backbone():
    """Build the backbone."""
    backbone_types = cfg.BACKBONE.TYPE.split('.')
    backbone = BACKBONES.get(backbone_types[0])()
    backbone_dims = backbone.out_dims
    neck = nn.Identity()
    if len(backbone_types) > 1:
        neck = NECKS.get(backbone_types[1])(backbone_dims)
    else:
        neck.out_dims = backbone_dims
    return backbone, neck


def build_detector(device=None, weights=None, training=False):
    """Create a detector instance.

    Parameters
    ----------
    device : int, optional
        The index of compute device.
    weights : str, optional
        The path of weight file.
    training : bool, optional, default=False
        Return a training detector or not.

    """
    model = DETECTORS.get(cfg.MODEL.TYPE)()
    if model is None:
        raise ValueError('Unknown detector: ' + cfg.MODEL.TYPE)
    if weights is not None:
        model.load_weights(weights, strict=True)
    if device is not None:
        model.to(device=get_device(device))
    if not training:
        model.eval()
        model.optimize_for_inference()
    return model
