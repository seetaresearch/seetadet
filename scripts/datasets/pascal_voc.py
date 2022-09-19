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
"""Prepare PASCAL VOC datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import codewithgpu
import cv2
import numpy as np
import xml.etree.ElementTree


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare PASCAL VOC datasets')
    parser.add_argument(
        '--rec',
        default=None,
        help='path to write record dataset')
    parser.add_argument(
        '--gt',
        default=None,
        help='path to write json dataset')
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


def make_example(img_file, xml_file):
    """Return the record example."""
    tree = xml.etree.ElementTree.parse(xml_file)
    filename = os.path.split(xml_file)[-1]
    objects = tree.findall('object')
    size = tree.find('size')
    example = {'id': filename.split('.')[0], 'object': []}
    with open(img_file, 'rb') as f:
        img_bytes = bytes(f.read())
    if size is not None:
        example['height'] = int(size.find('height').text)
        example['width'] = int(size.find('width').text)
        example['depth'] = int(size.find('depth').text)
    else:
        img = cv2.imdecode(np.frombuffer(img_bytes, 'uint8'), 3)
        example['height'], example['width'], example['depth'] = img.shape
    example['content'] = img_bytes
    for obj in objects:
        bbox = obj.find('bndbox')
        is_diff = 0
        if obj.find('difficult') is not None:
            is_diff = int(obj.find('difficult').text) == 1
        example['object'].append({
            'name': obj.find('name').text.strip(),
            'xmin': float(bbox.find('xmin').text),
            'ymin': float(bbox.find('ymin').text),
            'xmax': float(bbox.find('xmax').text),
            'ymax': float(bbox.find('ymax').text),
            'difficult': is_diff})
    return example


def write_dataset(args):
    """Write the record dataset."""
    assert len(args.splits) == len(args.images)
    assert len(args.splits) == len(args.annotations)
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
                'difficult': 'int64',
            }]
        }
    )

    # Scan all available entries.
    print('Scan entries...')
    entries = []
    for i, split in enumerate(args.splits):
        with open(split, 'r') as f:
            lines = f.readlines()
        for line in lines:
            filename = line.strip()
            img_file = os.path.join(args.images[i], filename + '.jpg')
            ann_file = os.path.join(args.annotations[i], filename + '.xml')
            entries.append((img_file, ann_file))

    # Parse and write into record file.
    print('Start Time:', time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
    start_time = time.time()
    for i, (img_file, xml_file) in enumerate(entries):
        if i > 0 and i % 2000 == 0:
            now_time = time.time()
            print('{} / {} in {:.2f} sec'.format(
                i, len(entries), now_time - start_time))
        writer.write(make_example(img_file, xml_file))
    now_time = time.time()
    print('{} / {} in {:.2f} sec'.format(
        len(entries), len(entries), now_time - start_time))
    writer.close()

    end_time = time.time()
    data_size = os.path.getsize(args.rec + '/00000.data') * 1e-6
    print('{} images take {:.2f} MB in {:.2f} sec.'
          .format(len(entries), data_size, end_time - start_time))


def write_json_dataset(args):
    """Write the json dataset."""
    categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']
    import subprocess
    scirpt = os.path.dirname(os.path.abspath(__file__)) + '/json_dataset.py'
    cmd = '{} {} '.format(sys.executable, scirpt)
    cmd += '--rec {} --gt {} '.format(args.rec, args.gt)
    cmd += '--categories {} '.format(' '.join(categories))
    return subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    args = parse_args()
    if args.rec is not None:
        write_dataset(args)
    if args.gt is not None:
        write_json_dataset(args)
