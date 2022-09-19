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
        default='',
        help='experiment dir')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='write metrics to tensorboard or not')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    coordinator = Coordinator(args.cfg_file, exp_dir=args.exp_dir)
    checkpoint, start_iter = coordinator.get_checkpoint()
    cfg.TRAIN.WEIGHTS = checkpoint or cfg.TRAIN.WEIGHTS

    # Setup the distributed environment.
    world_rank = dragon.distributed.get_rank()
    world_size = dragon.distributed.get_world_size()
    if cfg.NUM_GPUS != world_size:
        raise ValueError(
            'Excepted staring of {} processes, got {}.'
            .format(cfg.NUM_GPUS, world_size))

    # Setup the logging modules.
    logging.set_root(world_rank == 0)

    # Select the GPU depending on the rank of process.
    cfg.GPU_ID = [i for i in range(cfg.NUM_GPUS)][world_rank]

    # Fix the random seed for reproducibility.
    numpy.random.seed(cfg.RNG_SEED + world_rank)
    dragon.random.set_seed(cfg.RNG_SEED)

    # Inspect the dataset.
    dataset_size = build_dataset(cfg.TRAIN.DATASET).size
    logging.info('Dataset({}): {} images will be used to train.'
                 .format(cfg.TRAIN.DATASET, dataset_size))

    # Run training.
    logging.info('Checkpoints will be saved to `{:s}`'
                 .format(coordinator.path_at('checkpoints')))
    with dragon.distributed.new_group(
            ranks=[i for i in range(cfg.NUM_GPUS)],
            verbose=True).as_default():
        train_engine.run_train(
            coordinator, start_iter,
            enable_tensorboard=args.tensorboard)
