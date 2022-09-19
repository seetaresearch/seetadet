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
"""Build for training library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torch

from seetadet.core.config import cfg
from seetadet.core.engine import lr_scheduler


def build_optimizer(params, **kwargs):
    """Build the optimizer."""
    args = {'lr': cfg.SOLVER.BASE_LR,
            'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
            'clip_norm': cfg.SOLVER.CLIP_NORM,
            'grad_scale': 1.0 / cfg.SOLVER.LOSS_SCALE}
    optimizer = kwargs.pop('optimizer', cfg.SOLVER.OPTIMIZER)
    if optimizer == 'SGD':
        args['momentum'] = cfg.SOLVER.MOMENTUM
    args.update(kwargs)
    return getattr(torch.optim, optimizer)(params, **args)


def build_lr_scheduler(**kwargs):
    """Build the LR scheduler."""
    args = {'lr_max': cfg.SOLVER.BASE_LR,
            'lr_min': cfg.SOLVER.MIN_LR,
            'warmup_steps': cfg.SOLVER.WARM_UP_STEPS,
            'warmup_factor': cfg.SOLVER.WARM_UP_FACTOR}
    policy = kwargs.pop('policy', cfg.SOLVER.LR_POLICY)
    args.update(kwargs)
    if policy == 'steps_with_decay':
        return lr_scheduler.MultiStepLR(
            decay_steps=cfg.SOLVER.DECAY_STEPS,
            decay_gamma=cfg.SOLVER.DECAY_GAMMA, **args)
    elif policy == 'linear_decay':
        return lr_scheduler.LinearLR(
            decay_step=(cfg.SOLVER.DECAY_STEPS or [1])[0],
            max_steps=cfg.SOLVER.MAX_STEPS, **args)
    elif policy == 'cosine_decay':
        return lr_scheduler.CosineLR(
            decay_step=(cfg.SOLVER.DECAY_STEPS or [1])[0],
            max_steps=cfg.SOLVER.MAX_STEPS, **args)
    return lr_scheduler.ConstantLR(**args)


def build_tensorboard(log_dir):
    """Build the tensorboard."""
    try:
        from dragon.utils.tensorboard import tf
        from dragon.utils.tensorboard import TensorBoard
        # Avoid using of GPUs by TF API.
        if tf is not None:
            tf.config.set_visible_devices([], 'GPU')
        return TensorBoard(log_dir)
    except ImportError:
        return None
