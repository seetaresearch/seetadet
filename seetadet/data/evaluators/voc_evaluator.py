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
"""VOC dataset evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import numpy as np
import prettytable

from seetadet.data.build import EVALUATORS
from seetadet.data.evaluators.evaluator import Evaluator
from seetadet.data.evaluators.voc_eval import VOCeval


@EVALUATORS.register(['voc', 'voc2007', 'voc2010', 'voc2012'])
class VOCEvaluator(Evaluator):
    """Evaluator for Pascal VOC dataset."""

    def __init__(self, output_dir, classes, use_07_metric=False):
        eval_type = functools.partial(
            VOCeval, iouThrs=[0.5], use_07_metric=use_07_metric)
        super(VOCEvaluator, self).__init__(output_dir, classes, eval_type)

    def print_eval_results(self, coco_eval):
        metrics = collections.OrderedDict()
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            for k, name in zip(('ap', 'recall'), ('AP', 'AR')):
                for i, iou in enumerate(coco_eval.params.iouThrs):
                    name = '%s@[IoU=%s]' % (name, str(iou))
                    v = coco_eval.eval[k][i, cls_ind - 1]
                    if name not in metrics:
                        metrics[name] = []
                    metrics[name].append(v)
        class_table = prettytable.PrettyTable()
        summary_list = []
        for k, v in metrics.items():
            v = np.nan_to_num(v, nan=0)
            class_table.add_column(k, np.round(v * 100, 2))
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr, typeStr = 'Average Precision', '(AP)'
            if k.startswith('AR'):
                titleStr, typeStr = 'Average Recall', '(AR)'
            iouStr = '{:0.2f}'.format(float(k.split('IoU=')[-1][:-1]))
            summary_list.append(iStr.format(titleStr, typeStr, iouStr, 'all', -1, np.mean(v)))
        class_table.add_column('Class', self.classes[1:])
        print('Per class results:\n' + class_table.get_string(), '\n')
        print('Summary:\n' + '\n'.join(summary_list))
