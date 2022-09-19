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
"""Evaluation API on the Pascal VOC dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import itertools
import time

import numpy as np
from pycocotools import mask as maskUtils


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall."""
    if use_07_metric:
        # 11 point metric.
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # Correct AP calculation.
        # First append sentinel values at the end.
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # Compute the precision envelope.
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # To calculate area under PR curve, look for points.
        # where X axis (recall) changes value.
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # And sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class VOCeval(object):
    """Interface for evaluating detection via COCO object."""

    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox',
                 iouThrs=[0.5, 0.7], use_07_metric=False):
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.params = Params(iouType)
        self.params.iouThrs = iouThrs
        self.params.use_07_metric = use_07_metric
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        self.ious = {}

    def _prepare(self):
        p = self.params
        gts = self.cocoGt.loadAnns(
            self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(
            self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        self._gts = collections.defaultdict(list)
        self._dts = collections.defaultdict(list)
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.eval = {}

    def evaluate(self):
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        self._prepare()
        self.ious = {(imgId, catId): self.computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in p.catIds}
        self.evalImgs = [self.evaluateImg(imgId, catId)
                         for catId in p.catIds for imgId in p.imgIds]
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def accumulate(self, p=None):
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        if p is None:
            p = self.params
        print('VOC07 metric? ' + ('Yes' if p.use_07_metric else 'No'))
        T, K, I = len(p.iouThrs), len(p.catIds), len(p.imgIds)
        recall, ap = np.zeros((T, K)), np.zeros((T, K))
        for k in range(K):
            E = [self.evalImgs[k * I + i] for i in range(I)]
            E = [e for e in E if e is not None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'] for e in E])
            inds = np.argsort(-dtScores)
            dtm = np.concatenate([e['dtMatches'] for e in E], axis=1)[:, inds]
            dtIg = np.concatenate([e['dtIgnore'] for e in E], axis=1)[:, inds]
            gtIg = np.concatenate([e['gtIgnore'] for e in E])
            npig = np.count_nonzero(gtIg == 0)
            if npig == 0:
                continue
            tps = np.logical_and(dtm, np.logical_not(dtIg))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                nd = len(tp)
                rc = tp / npig
                pr = tp / np.maximum(tp + fp, np.spacing(1))
                recall[t, k] = rc[-1] if nd else 0
                ap[t, k] = voc_ap(rc, pr, use_07_metric=p.use_07_metric)
        self.eval = {'counts': [T, K],
                     'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     'ap': ap, 'recall': recall}
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')
        iscrowd = [int(o['iscrowd']) for o in gt]
        return maskUtils.iou(d, g, iscrowd)

    def evaluateImg(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None
        for g in gt:
            g['_ignore'] = g['ignore']
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind]
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = (self.ious[imgId, catId][:, gtind]
                if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId])
        T, G, D = len(p.iouThrs), len(gt), len(dt)
        gtm, dtm = np.zeros((T, G)), np.zeros((T, D))
        gtIg, dtIg = np.array([g['_ignore'] for g in gt]), np.zeros((T, D))
        for (tind, iou), (dind, d) in itertools.product(
                enumerate(p.iouThrs), enumerate(dt)):
            m = -1
            for gind, g in enumerate(gt):
                if gtm[tind, gind] > 0 and not iscrowd[gind]:
                    continue
                if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                    break
                if ious[dind, gind] <= iou:
                    continue
                m = gind
            if m == -1:
                continue
            dtIg[tind, dind] = gtIg[m]
            dtm[tind, dind] = gt[m]['id']
            gtm[tind, m] = d['id']
        return {'image_id': imgId,
                'category_id': catId,
                'dtMatches': dtm,
                'dtScores': [d['score'] for d in dt],
                'gtIgnore': gtIg,
                'dtIgnore': dtIg}


class Params(object):
    """Params for evaluation API."""

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        self.iouThrs = [0.5]
        self.use_07_metric = False

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
