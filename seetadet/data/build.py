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
"""Build for data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.core.config import cfg
from seetadet.core.registry import Registry

LOADERS = Registry('loaders')
DATASETS = Registry('datasets')
EVALUATORS = Registry('evaluators')
ANCHOR_SAMPLERS = Registry('anchor_samplers')


def build_anchor_sampler():
    """Build the anchor sampler."""
    return ANCHOR_SAMPLERS.try_get(cfg.MODEL.TYPE)()


def build_dataset(path):
    """Build the dataset."""
    keys = path.split('://')
    if len(keys) >= 2:
        return DATASETS.get(keys[0])(keys[1])
    return DATASETS.get('default')(path)


def build_loader_train(**kwargs):
    """Build the train loader."""
    args = {'dataset': cfg.TRAIN.DATASET,
            'batch_size': cfg.TRAIN.IMS_PER_BATCH,
            'num_workers': cfg.TRAIN.NUM_WORKERS,
            'shuffle': True, 'contiguous': True}
    args.update(kwargs)
    return LOADERS.get(cfg.TRAIN.LOADER)(**args)


def build_loader_test(**kwargs):
    """Build the test loader."""
    args = {'dataset': cfg.TEST.DATASET,
            'batch_size': cfg.TEST.IMS_PER_BATCH,
            'shuffle': False, 'contiguous': False}
    args.update(kwargs)
    return LOADERS.get(cfg.TEST.LOADER)(**args)


def build_evaluator(output_dir, **kwargs):
    """Build the evaluator."""
    evaluator_type = cfg.TEST.EVALUATOR
    if not evaluator_type:
        return None
    args = {'output_dir': output_dir,
            'classes': cfg.MODEL.CLASSES}
    if evaluator_type == 'voc2007':
        args['use_07_metric'] = True
    args.update(kwargs)
    evaluator = EVALUATORS.get(evaluator_type)(**args)
    ann_file = cfg.TEST.JSON_DATASET
    if ann_file:
        evaluator.load_annotations(ann_file)
    return evaluator
