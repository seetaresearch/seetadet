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
"""Train a detection network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import dragon
import numpy

from seetadet.core.config import cfg
from seetadet.core.coordinator import Coordinator
from seetadet.core.engine import train_engine
from seetadet.data.build import build_dataset
from seetadet.utils import logging


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Train a detection network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        default=None,
        help='config file')
    parser.add_argument(
        '--exp_dir',
        default=None,
        help='experiment dir')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='write metrics to tensorboard or not')
    return parser.parse_args()


def run_distributed(args, coordinator):
    """Run distributed training."""
    import subprocess
    cmd = 'mpirun --allow-run-as-root -n {} --bind-to none '.format(cfg.NUM_GPUS)
    cmd += '{} {}'.format(sys.executable, 'distributed/train.py')
    cmd += ' --cfg {}'.format(os.path.abspath(args.cfg_file))
    cmd += ' --exp_dir {}'.format(coordinator.exp_dir)
    cmd += ' --tensorboard' if args.tensorboard else ''
    return subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    args = parse_args()
    logging.info('Called with args:\n' + str(args))

    coordinator = Coordinator(args.cfg_file, args.exp_dir)
    checkpoint, start_iter = coordinator.get_checkpoint()
    cfg.TRAIN.WEIGHTS = checkpoint or cfg.TRAIN.WEIGHTS
    logging.info('Using config:\n' + str(cfg))

    if cfg.NUM_GPUS > 1:
        # Run a distributed task.
        run_distributed(args, coordinator)
    else:
        # Fix the random seed for reproducibility.
        numpy.random.seed(cfg.RNG_SEED)
        dragon.random.set_seed(cfg.RNG_SEED)

        # Inspect the dataset.
        dataset_size = build_dataset(cfg.TRAIN.DATASET).size
        logging.info('Dataset({}): {} images will be used to train.'
                     .format(cfg.TRAIN.DATASET, dataset_size))

        # Run training.
        logging.info('Checkpoints will be saved to `{:s}`'
                     .format(coordinator.path_at('checkpoints')))
        train_engine.run_train(coordinator, start_iter,
                               enable_tensorboard=args.tensorboard)
