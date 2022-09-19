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
"""Base evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import numpy as np
from pycocotools.coco import COCO

from seetadet.data.build import build_loader_test
from seetadet.utils import logging
from seetadet.utils.mask import encode_masks
from seetadet.utils.mask import paste_masks


class Evaluator(object):
    """Evaluator using COCO json dataset format."""

    def __init__(self, output_dir, classes, eval_type=None):
        self.output_dir = output_dir
        self.classes = classes
        self.num_classes = len(self.classes)
        self.class_to_cat_id = dict(zip(self.classes, range(self.num_classes)))
        self.eval_type = eval_type
        self.cocoGt = None
        self.loader = build_loader_test()
        self.num_images = self.loader.dataset_size
        self.cached_inputs = []
        self.records = collections.OrderedDict()

    def eval_bbox(self, boxes):
        """Evaluate bbox results."""
        if len(self.cocoGt.dataset['annotations']) == 0:
            logging.info('No annotations. Skip evaluation.')
            return
        self.verify_records()
        res_file = self.write_bbox_results(boxes)
        cocoDt = self.cocoGt.loadRes(res_file)
        coco_eval = self.eval_type(self.cocoGt, cocoDt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self.print_eval_results(coco_eval)

    def eval_segm(self, boxes, masks):
        """Evaluate segmentation results."""
        if len(self.cocoGt.dataset['annotations']) == 0:
            logging.info('No annotations. Skip evaluation.')
            return
        self.verify_records()
        res_file = self.write_segm_results(boxes, masks)
        cocoDt = self.cocoGt.loadRes(res_file)
        coco_eval = self.eval_type(self.cocoGt, cocoDt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self.print_eval_results(coco_eval)

    def get_image(self):
        """Return the next image for evaluation."""
        if len(self.cached_inputs) == 0:
            inputs = self.loader()
            for i, img_meta in enumerate(inputs['img_meta']):
                self.cached_inputs.append({
                    'img': inputs['img'][i],
                    'objects': inputs['objects'][i],
                    'id': img_meta['id'],
                    'height': img_meta['height'],
                    'width': img_meta['width']})
        inputs = self.cached_inputs.pop(0)
        img_id, img = inputs.pop('id'), inputs.pop('img')
        self.records[img_id] = inputs
        return img_id, img

    def load_annotations(self, ann_file=None):
        """Load annotations."""
        self.cocoGt = COCO(ann_file)
        if len(self.cocoGt.dataset) > 0:
            self.class_to_cat_id = dict((v['name'], v['id'])
                                        for v in self.cocoGt.cats.values())

    def verify_records(self):
        """Verify loaded records."""
        if len(self.records) != self.num_images:
            raise RuntimeError(
                'Mismatched number of records and images. ({} vs. {}).'
                '\nCheck if existing duplicate image ids.'
                .format(len(self.records), self.num_images))
        if self.cocoGt is None:
            ann_file = self.write_annotations(self.records, self.output_dir)
            self.load_annotations(ann_file)

    def print_eval_results(self, coco_eval):
        """Print the evaluation results."""

    def bbox_results_one_category(self, boxes, cat_id):
        """Write bbox results of a specific category."""
        results = []
        for i, img_id in enumerate(self.records.keys()):
            dets = boxes[i].astype('float64')
            if len(dets) == 0:
                continue
            xs, ys = dets[:, 0], dets[:, 1]
            ws, hs = dets[:, 2] - xs, dets[:, 3] - ys
            scores = dets[:, -1]
            results.extend([{
                'image_id': self.get_image_id(img_id),
                'category_id': cat_id,
                'bbox': [xs[j], ys[j], ws[j], hs[j]],
                'score': scores[j],
            } for j in range(dets.shape[0])])
        return results

    def segm_results_one_category(self, boxes, masks, cat_id):
        """Write segm results of a specific category."""
        results = []
        for i, (img_id, rec) in enumerate(self.records.items()):
            dets = boxes[i]
            if len(dets) == 0:
                continue
            scores = dets[:, -1]
            rles = encode_masks(paste_masks(
                masks[i], dets, (rec['height'], rec['width'])))
            results.extend([{
                'image_id': self.get_image_id(img_id),
                'category_id': cat_id,
                'segmentation': rles[j],
                'score': float(scores[j]),
            } for j in range(dets.shape[0])])
        return results

    def write_bbox_results(self, boxes):
        """Write bbox results."""
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'
                  .format(cls, cls_ind, self.num_classes - 1))
            results.extend(self.bbox_results_one_category(
                boxes[cls_ind], self.class_to_cat_id[cls]))
        res_file = self.get_res_file(type='bbox')
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as f:
            json.dump(results, f)
        return res_file

    def write_segm_results(self, boxes, masks):
        """Write segm results."""
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'
                  .format(cls, cls_ind, self.num_classes - 1))
            results.extend(self.segm_results_one_category(
                boxes[cls_ind], masks[cls_ind], self.class_to_cat_id[cls]))
        res_file = self.get_res_file(type='segm')
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)
        return res_file

    def write_annotations(self):
        """Write annotations."""
        dataset = {'images': [], 'categories': [], 'annotations': []}
        for img_id, rec in self.records.items():
            dataset['images'].append({
                'id': self.get_image_id(img_id),
                'height': rec['height'], 'width': rec['width']})
        for cls in self.classes:
            if cls == '__background__':
                continue
            dataset['categories'].append({
                'name': cls, 'id': self.class_to_cat_id[cls]})
        for img_id, rec in self.records.items():
            img_size = (rec['height'], rec['width'])
            for obj in rec['objects']:
                x, y = obj['bbox'][0], obj['bbox'][1]
                w, h = obj['bbox'][2] - x, obj['bbox'][3] - y
                dataset['annotations'].append({
                    'id': str(len(dataset['annotations'])),
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': obj['difficult'],
                    'image_id': self.get_image_id(img_id),
                    'category_id': self.class_to_cat_id[obj['name']]})
                if 'mask' in obj:
                    segm = {'size': img_size, 'counts': obj['mask']}
                    dataset['annotations'][-1]['segmentation'] = segm
                elif 'polygons' in obj:
                    segm = []
                    for poly in obj['polygons']:
                        if isinstance(poly, np.ndarray):
                            poly = poly.tolist()
                        segm.append(poly)
                    dataset['annotations'][-1]['segmentation'] = segm
        ann_file = self.get_ann_file()
        print('Writing annotations json to {}'.format(ann_file))
        with open(ann_file, 'w') as f:
            json.dump(dataset, f)
        return ann_file

    def get_ann_file(self):
        """Return the ann filename."""
        filename = 'annotations.json'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return os.path.join(self.output_dir, filename)

    def get_res_file(self, type='bbox'):
        """Return the result filename."""
        prefix = ''
        if type == 'bbox':
            prefix = 'detections'
        elif type == 'segm':
            prefix = 'segmentations'
        elif type == 'kpt':
            prefix = 'keypoints'
        filename = prefix + '.json'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return os.path.join(self.output_dir, filename)

    @staticmethod
    def get_image_id(image_name):
        """Return the image name from the id."""
        image_id = image_name.split('_')[-1].split('.')[0]
        try:
            return int(image_id)
        except ValueError:
            return image_name
