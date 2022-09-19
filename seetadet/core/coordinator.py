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
"""Experiment coordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import time

import numpy as np

from seetadet.core.config import cfg
from seetadet.utils import logging


class Coordinator(object):
    """Manage the unique experiments."""

    def __init__(self, cfg_file, exp_dir=None):
        cfg.merge_from_file(cfg_file)
        if exp_dir is None:
            name = time.strftime('%Y%m%d_%H%M%S',
                                 time.localtime(time.time()))
            exp_dir = '../experiments/{}'.format(name)
            if not osp.exists(exp_dir):
                os.makedirs(exp_dir)
        else:
            if not osp.exists(exp_dir):
                raise ValueError('Invalid experiment dir: ' + exp_dir)
        self.exp_dir = exp_dir

    def path_at(self, file, auto_create=True):
        try:
            path = osp.abspath(osp.join(self.exp_dir, file))
            if auto_create and not osp.exists(path):
                os.makedirs(path)
        except OSError:
            path = osp.abspath(osp.join('/tmp', file))
            if auto_create and not osp.exists(path):
                os.makedirs(path)
        return path

    def get_checkpoint(self, step=None, last_idx=1, wait=False):
        path = self.path_at('checkpoints')

        def locate(last_idx=None):
            files = os.listdir(path)
            files = list(filter(lambda x: '_iter_' in x and
                                          x.endswith('.pkl'), files))
            file_steps = []
            for i, file in enumerate(files):
                file_step = int(file.split('_iter_')[-1].split('.')[0])
                if step == file_step:
                    return osp.join(path, files[i]), file_step
                file_steps.append(file_step)
            if step is None:
                if len(files) == 0:
                    return None, 0
                if last_idx > len(files):
                    return None, 0
                file = files[np.argsort(file_steps)[-last_idx]]
                file_step = file_steps[np.argsort(file_steps)[-last_idx]]
                return osp.join(path, file), file_step
            return None, 0

        file, file_step = locate(last_idx)
        while file is None and wait:
            logging.info('Wait for checkpoint at {}.'.format(step))
            time.sleep(10)
            file, file_step = locate(last_idx)
        return file, file_step
