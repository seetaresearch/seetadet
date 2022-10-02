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
"""Prepare JSON datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

import codewithgpu


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare JSON datasets')
    parser.add_argument(
        '--rec',
        default=None,
        help='path to read record')
    parser.add_argument(
        '--gt',
        default=None,
        help='path to write json ground-truth')
    parser.add_argument(
        '--categories',
        nargs='+',
        type=str,
        default=None,
        help='dataset object categories')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_image_id(image_name):
    image_id = image_name.split('_')[-1].split('.')[0]
    try:
        return int(image_id)
    except ValueError:
        return image_name


def write_dataset(args):
    dataset = {'images': [], 'categories': [], 'annotations': []}
    record_dataset = codewithgpu.RecordDataset(args.rec)
    cat_to_cat_id = dict(zip(args.categories,
                             range(1, len(args.categories) + 1)))
    print('Writing json dataset to {}'.format(args.gt))
    for cat in args.categories:
        dataset['categories'].append({
            'name': cat, 'id': cat_to_cat_id[cat]})
    for example in record_dataset:
        image_id = get_image_id(example['id'])
        dataset['images'].append({
            'id': image_id, 'height': example['height'],
            'width': example['width']})
        for obj in example['object']:
            if 'x2' in obj:
                x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            elif 'xmin' in obj:
                x1, y1, x2, y2 = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
            else:
                x1, y1, x2, y2 = obj['bbox']
            w, h = x2 - x1, y2 - y1
            dataset['annotations'].append({
                'id': str(len(dataset['annotations'])),
                'bbox': [x1, y1, w, h],
                'area': w * h,
                'iscrowd': obj.get('difficult', 0),
                'image_id': image_id,
                'category_id': cat_to_cat_id[obj['name']]})
    with open(args.gt, 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    args = parse_args()
    if args.rec is None or not os.path.exists(args.rec):
        raise ValueError('Specify the prepared record dataset.')
    if args.gt is None:
        raise ValueError('Specify the path to write json dataset.')
    write_dataset(args)
