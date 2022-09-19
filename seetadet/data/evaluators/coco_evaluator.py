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
"""COCO dataset evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import prettytable
from pycocotools.cocoeval import COCOeval

from seetadet.data.build import EVALUATORS
from seetadet.data.evaluators.evaluator import Evaluator


@EVALUATORS.register('coco')
class COCOEvaluator(Evaluator):
    """Evaluator for MS COCO dataset."""

    def __init__(self, output_dir, classes):
        super(COCOEvaluator, self).__init__(output_dir, classes, COCOeval)

    def print_eval_results(self, coco_eval):
        def get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind
        ind_lo = get_thr_ind(coco_eval, 0.5)
        ind_hi = get_thr_ind(coco_eval, 0.95)
        # Precision: (iou, recall, cls, area range, max dets)
        # Recall: (iou, cls, area range, max dets)
        # Area range index 0: all area ranges
        # Max dets index 2: 100 per image
        all_prec = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        all_recall = coco_eval.eval['recall'][ind_lo:(ind_hi + 1), :, 0, 2]
        metrics = collections.OrderedDict([
            ('AP@[IoU=0.5:0.95]', []), ('AR@[IoU=0.5:0.95]', [])])
        class_table = prettytable.PrettyTable()
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            ap = np.mean(all_prec[:, :, cls_ind - 1])  # (iou, recall, cls)
            recall = np.mean(all_recall[:, cls_ind - 1])  # (iou, cls)
            metrics['AP@[IoU=0.5:0.95]'].append(ap)
            metrics['AR@[IoU=0.5:0.95]'].append(recall)
        for k, v in metrics.items():
            v = np.nan_to_num(v, nan=0)
            class_table.add_column(k, np.round(v * 100, 2))
        class_table.add_column('Class', self.classes[1:])
        print('Per class results:\n' + class_table.get_string(), '\n')
        print('Summary:')
        coco_eval.summarize()
