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
"""Training engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os

from dragon.vm import torch

from seetadet.core.config import cfg
from seetadet.core.engine.build import build_lr_scheduler
from seetadet.core.engine.build import build_optimizer
from seetadet.core.engine.build import build_tensorboard
from seetadet.core.engine.utils import count_params
from seetadet.core.engine.utils import get_device
from seetadet.core.engine.utils import get_param_groups
from seetadet.data.build import build_loader_train
from seetadet.models.build import build_detector
from seetadet.utils import logging
from seetadet.utils import profiler


class Trainer(object):
    """Schedule the iterative model training."""

    def __init__(self, coordinator, start_iter=0):
        # Build loader.
        self.loader = build_loader_train()
        # Build model.
        self.model = build_detector(training=True)
        self.model.load_weights(cfg.TRAIN.WEIGHTS, strict=start_iter > 0)
        self.model.to(device=get_device(cfg.GPU_ID))
        if cfg.MODEL.PRECISION.lower() == 'float16':
            self.model.half()
        # Build optimizer.
        self.loss_scale = cfg.SOLVER.LOSS_SCALE
        param_groups_getter = get_param_groups
        if cfg.SOLVER.LAYER_LR_DECAY < 1.0:
            lr_scale_getter = functools.partial(
                self.model.backbone.get_lr_scale,
                decay=cfg.SOLVER.LAYER_LR_DECAY)
            param_groups_getter = functools.partial(
                param_groups_getter, lr_scale_getter=lr_scale_getter)
        self.optimizer = build_optimizer(param_groups_getter(self.model))
        self.scheduler = build_lr_scheduler()
        # Build monitor.
        self.coordinator = coordinator
        self.metrics = collections.OrderedDict()
        self.board = None

    @property
    def iter(self):
        return self.scheduler._step_count

    def snapshot(self):
        """Save the checkpoint of current iterative step."""
        f = cfg.SOLVER.SNAPSHOT_PREFIX
        f += '_iter_{}.pkl'.format(self.iter)
        f = os.path.join(self.coordinator.path_at('checkpoints'), f)
        if logging.is_root() and not os.path.exists(f):
            torch.save(self.model.state_dict(), f, pickle_protocol=4)
            logging.info('Wrote snapshot to: {:s}'.format(f))

    def add_metrics(self, stats):
        """Add or update the metrics."""
        for k, v in stats['metrics'].items():
            if k not in self.metrics:
                self.metrics[k] = profiler.SmoothedValue()
            self.metrics[k].update(v)

    def display_metrics(self, stats):
        """Send metrics to the monitor."""
        logging.info('Iteration %d, lr = %.8f, time = %.2fs'
                     % (stats['iter'], stats['lr'], stats['time']))
        for k, v in self.metrics.items():
            logging.info(' ' * 4 + 'Train net output({}): {:.4f} ({:.4f})'
                         .format(k, stats['metrics'][k], v.average()))
        if self.board is not None:
            self.board.scalar_summary('lr', stats['lr'], stats['iter'])
            self.board.scalar_summary('time', stats['time'], stats['iter'])
            for k, v in self.metrics.items():
                self.board.scalar_summary(k, v.average(), stats['iter'])

    def step(self):
        stats = {'iter': self.iter}
        metrics = collections.defaultdict(float)
        # Run forward.
        timer = profiler.Timer().tic()
        inputs = self.loader()
        outputs, losses = self.model(inputs), []
        for k, v in outputs.items():
            if 'loss' in k:
                if isinstance(v, (tuple, list)):
                    losses.append(sum(v[1:], v[0]))
                    metrics.update(dict(('stage%d_' % (i + 1) + k, float(x))
                                        for i, x in enumerate(v)))
                else:
                    losses.append(v)
                    metrics[k] += float(v)
        # Run backward.
        losses = sum(losses[1:], losses[0])
        if self.loss_scale != 1.0:
            losses *= self.loss_scale
        losses.backward()
        # Apply update.
        stats['lr'] = self.scheduler.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = stats['lr'] * group.get('lr_scale', 1.0)
        self.optimizer.step()
        self.scheduler.step()
        stats['time'] = timer.toc()
        stats['metrics'] = collections.OrderedDict(sorted(metrics.items()))
        return stats

    def train_model(self, start_iter=0):
        """Network training loop."""
        timer = profiler.Timer()
        max_steps = cfg.SOLVER.MAX_STEPS
        display_every = cfg.SOLVER.DISPLAY
        progress_every = 10 * display_every
        snapshot_every = cfg.SOLVER.SNAPSHOT_EVERY
        self.scheduler._step_count = start_iter
        while self.iter < max_steps:
            with timer.tic_and_toc():
                stats = self.step()
            self.add_metrics(stats)
            if stats['iter'] % display_every == 0:
                self.display_metrics(stats)
            if self.iter % progress_every == 0:
                logging.info(profiler.get_progress(timer, self.iter, max_steps))
            if self.iter % snapshot_every == 0:
                self.snapshot()
                self.metrics.clear()


def run_train(coordinator, start_iter=0, enable_tensorboard=False):
    """Start a network training task."""
    trainer = Trainer(coordinator, start_iter=start_iter)
    if enable_tensorboard and logging.is_root():
        trainer.board = build_tensorboard(coordinator.path_at('logs'))
    logging.info('#Params: %.2fM' % count_params(trainer.model))
    logging.info('Start training...')
    trainer.train_model(start_iter)
    trainer.snapshot()
