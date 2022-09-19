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
"""Test a detection network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os

from seetadet.core.config import cfg
from seetadet.core.coordinator import Coordinator
from seetadet.core.engine import test_engine
from seetadet.data.build import build_dataset
from seetadet.utils import logging


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Test a detection network')
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
        '--model_dir',
        default='',
        help='model dir')
    parser.add_argument(
        '--gpu',
        nargs='+',
        type=int,
        default=None,
        help='index of GPUs to use')
    parser.add_argument(
        '--iter',
        nargs='+',
        type=int,
        default=None,
        help='iteration step of checkpoints')
    parser.add_argument(
        '--last',
        type=int,
        default=1,
        help='last N checkpoints')
    parser.add_argument(
        '--read_every',
        type=int,
        default=100,
        help='read every-n images for testing')
    parser.add_argument(
        '--vis',
        type=float,
        default=0,
        help='score threshold for visualization')
    parser.add_argument(
        '--precision',
        default='',
        help='compute precision')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='set cudnn deterministic or not')
    return parser.parse_args()


def find_weights(args, coordinator):
    """Return the weights for testing."""
    weights_list = []
    if args.model_dir:
        for file in os.listdir(args.model_dir):
            if not file.endswith('.pkl'):
                continue
            weights_list.append(os.path.join(args.model_dir, file))
        return weights_list
    if args.iter is not None:
        for iter in args.iter:
            checkpoint, _ = coordinator.get_checkpoint(iter, wait=True)
            weights_list.append(checkpoint)
        return weights_list
    for i in range(1, args.last + 1):
        checkpoint, _ = coordinator.get_checkpoint(last_idx=i)
        if checkpoint is None:
            break
        weights_list.append(checkpoint)
    return weights_list


if __name__ == '__main__':
    args = parse_args()
    logging.info('Called with args:\n' + str(args))

    coordinator = Coordinator(args.cfg_file, args.exp_dir or args.model_dir)
    cfg.MODEL.PRECISION = args.precision or cfg.MODEL.PRECISION
    logging.info('Using config:\n' + str(cfg))

    # Inspect dataset.
    dataset_size = build_dataset(cfg.TEST.DATASET).size
    logging.info('Dataset({}): {} images will be used to test.'
                 .format(cfg.TEST.DATASET, dataset_size))

    # Run testing.
    for weights in find_weights(args, coordinator):
        weights_name = os.path.splitext(os.path.basename(weights))[0]
        output_dir = coordinator.path_at('results/' + weights_name)
        logging.info('Results will be saved to ' + output_dir)
        vis_output_dir = None
        if args.vis > 0:
            vis_output_dir = coordinator.path_at('visualizations/' + weights_name)
            logging.info('Visualizations will be saved to ' + vis_output_dir)
        process = multiprocessing.Process(
            target=test_engine.run_test,
            kwargs={'test_cfg': cfg,
                    'weights': weights,
                    'output_dir': output_dir,
                    'devices': args.gpu,
                    'deterministic': args.deterministic,
                    'read_every': args.read_every,
                    'vis_thresh': args.vis,
                    'vis_output_dir': vis_output_dir})
        process.start()
        process.join()
