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
"""Prepare MS COCO datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import codewithgpu
from pycocotools.coco import COCO
from pycocotools.mask import frPyObjects


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare MS COCO datasets')
    parser.add_argument(
        '--rec',
        default=None,
        help='path to write record dataset')
    parser.add_argument(
        '--images',
        nargs='+',
        type=str,
        default=None,
        help='path of images folder')
    parser.add_argument(
        '--annotations',
        nargs='+',
        type=str,
        default=None,
        help='path of annotations folder')
    parser.add_argument(
        '--splits',
        nargs='+',
        type=str,
        default=None,
        help='path of split file')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_example(img_id, img_file, cocoGt):
    """Return the record example."""
    img_meta = cocoGt.imgs[img_id]
    img_anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[img_id]))
    cat_id_to_cat = dict((v['id'], v['name'])
                         for v in cocoGt.cats.values())
    with open(img_file, 'rb') as f:
        img_bytes = bytes(f.read())
    height, width = img_meta['height'], img_meta['width']
    example = {'id': str(img_id), 'height': height, 'width': width,
               'depth': 3, 'content': img_bytes, 'object': []}
    for ann in img_anns:
        x1 = float(max(0, ann['bbox'][0]))
        y1 = float(max(0, ann['bbox'][1]))
        x2 = float(min(width, x1 + max(0, ann['bbox'][2])))
        y2 = float(min(height, y1 + max(0, ann['bbox'][3])))
        mask, polygons = b'', []
        segm = ann.get('segmentation', None)
        if segm is not None and isinstance(segm, list):
            for p in ann['segmentation']:
                if len(p) < 6:
                    print('Remove Invalid segm.')
            # Valid polygons have >= 3 points, so require >= 6 coordinates
            polygons = [p for p in ann['segmentation'] if len(p) >= 6]
        elif segm is not None:
            # Crowd masks.
            # Some are encoded with wrong height or width.
            # Do not use them or decoding error is inevitable.
            rle = frPyObjects(ann['segmentation'], height, width)
            assert type(rle) == dict
            mask = rle['counts']
        example['object'].append({
            'name': cat_id_to_cat[ann['category_id']],
            'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
            'mask': mask, 'polygons': polygons,
            'difficult': ann.get('iscrowd', 0)})
    return example


def write_dataset(args):
    assert len(args.images) == len(args.annotations)
    if os.path.exists(args.rec):
        raise ValueError('The record path is already exist.')
    os.makedirs(args.rec)
    print('Write record dataset to {}'.format(args.rec))

    writer = codewithgpu.RecordWriter(
        path=args.rec,
        features={
            'id': 'string',
            'content': 'bytes',
            'height': 'int64',
            'width': 'int64',
            'depth': 'int64',
            'object': [{
                'name': 'string',
                'xmin': 'float64',
                'ymin': 'float64',
                'xmax': 'float64',
                'ymax': 'float64',
                'mask': 'bytes',
                'polygons': [['float64']],
                'difficult': 'int64',
            }]
        }
    )

    # Scan all available entries.
    print('Scan entries...')
    entries, cocoGts = [], []
    for ann_file in args.annotations:
        cocoGts.append(COCO(ann_file))
    if args.splits is not None:
        assert len(args.splits) == len(args.images)
        for i, split in enumerate(args.splits):
            f = open(split, 'r')
            for line in f.readlines():
                filename = line.strip()
                img_id = int(filename)
                img_file = os.path.join(args.images[i], filename + '.jpg')
                entries.append((img_id, img_file, cocoGts[i]))
            f.close()
    else:
        for i, cocoGt in enumerate(cocoGts):
            for info in cocoGt.imgs.values():
                img_id = info['id']
                img_file = os.path.join(args.images[i], info['file_name'])
                entries.append((img_id, img_file, cocoGts[i]))

    print('Start Time:', time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
    start_time = time.time()
    for i, entry in enumerate(entries):
        if i > 0 and i % 2000 == 0:
            now_time = time.time()
            print('{} / {} in {:.2f} sec'.format(
                i, len(entries), now_time - start_time))
        writer.write(make_example(*entry))
    now_time = time.time()
    print('{} / {} in {:.2f} sec'.format(
        len(entries), len(entries), now_time - start_time))
    writer.close()

    end_time = time.time()
    data_size = os.path.getsize(args.rec + '/00000.data') * 1e-6
    print('{} images take {:.2f} MB in {:.2f} sec.'
          .format(len(entries), data_size, end_time - start_time))


if __name__ == '__main__':
    args = parse_args()
    if args.rec is not None:
        write_dataset(args)
