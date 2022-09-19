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
"""Export a detection network into the onnx model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import dragon.vm.torch as torch
import numpy as np

from seetadet.core.config import cfg
from seetadet.core.coordinator import Coordinator
from seetadet.models.build import build_detector
from seetadet.ops import onnx as _  # noqa
from seetadet.utils import logging


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Export a detection network into the onnx model')
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
        type=int,
        default=0,
        help='index of GPU to use')
    parser.add_argument(
        '--iter',
        type=int,
        default=None,
        help='checkpoint of given step')
    parser.add_argument(
        '--input_shape',
        nargs='+',
        type=int,
        default=(1, 512, 512, 3),
        help='input image shape')
    parser.add_argument(
        '--opset',
        type=int,
        default=None,
        help='opset version to export')
    parser.add_argument(
        '--check_model',
        type=bool,
        default=True,
        help='check the model validation or not')
    return parser.parse_args()


def find_weights(args, coordinator):
    """Return the weights for exporting."""
    weights_list = []
    if args.model_dir:
        for file in os.listdir(args.model_dir):
            if not file.endswith('.pkl'):
                continue
            weights_list.append(os.path.join(args.model_dir, file))
    else:
        checkpoint, _ = coordinator.get_checkpoint(args.iter)
        weights_list.append(checkpoint)
    return weights_list[0]


def get_dummay_inputs(args):
    n, h, w, c = args.input_shape
    im_batch = torch.zeros(n, h, w, c, dtype='uint8')
    im_info = torch.tensor([[h, w, 1., 1.] for _ in range(n)], dtype='float32')
    strides = [2 ** x for x in range(cfg.FPN.MIN_LEVEL, cfg.FPN.MAX_LEVEL + 1)]
    strides = np.array(strides)[:, None]
    grid_shapes = np.stack([[h, w]] * len(strides))
    grid_shapes = (grid_shapes - 1) // strides + 1
    grid_info = torch.tensor(grid_shapes, dtype='int64')
    return {'img': im_batch, 'im_info': im_info, 'grid_info': grid_info}


if __name__ == '__main__':
    args = parse_args()
    logging.info('Called with args:\n' + str(args))

    coordinator = Coordinator(args.cfg_file, args.exp_dir or args.model_dir)
    logging.info('Using config:\n' + str(cfg))

    # Run exporting.
    weights = find_weights(args, coordinator)
    weights_name = os.path.splitext(os.path.basename(weights))[0]
    output_dir = args.model_dir or coordinator.path_at('exports')
    logging.info('Exports will be saved to ' + output_dir)
    detector = build_detector(args.gpu, weights)
    inputs = get_dummay_inputs(args)
    torch.onnx.export(
        model=detector,
        args=inputs,
        f=os.path.join(output_dir, weights_name + '.onnx'),
        verbose=True,
        opset_version=args.opset,
        enable_onnx_checker=args.check_model,
    )
