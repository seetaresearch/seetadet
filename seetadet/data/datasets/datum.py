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
"""Annotated datum."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


class AnnotatedDatum(object):
    """Wrapper for annotated datum."""

    def __init__(self, example):
        self._example = example
        self._img = None

    @property
    def id(self):
        """Return the example id."""
        return self._example['id']

    @property
    def height(self):
        """Return the image height."""
        return self._example['height']

    @property
    def width(self):
        """Return the image width."""
        return self._example['width']

    @property
    def img(self):
        """Return the image array."""
        if self._img is None:
            img_bytes = np.frombuffer(self._example['content'], 'uint8')
            self._img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        return self._img

    @property
    def objects(self):
        """Return the annotated objects."""
        objects = []
        for obj in self._example['object']:
            mask = obj.get('mask', None)
            polygons = obj.get('polygons', None)
            if 'x3' in obj:
                poly = np.array([obj['x1'], obj['y1'],
                                 obj['x2'], obj['y2'],
                                 obj['x3'], obj['y3'],
                                 obj['x4'], obj['y4']], 'float32')
                x, y, w, h = cv2.boundingRect(poly.reshape((-1, 2)))
                bbox = [x, y, x + w, y + h]
                polygons = [poly]
            elif 'x2' in obj:
                bbox = [obj['x1'], obj['y1'], obj['x2'], obj['y2']]
            elif 'xmin' in obj:
                bbox = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
            else:
                bbox = obj['bbox']
            objects.append({'name': obj['name'],
                            'bbox': bbox,
                            'difficult': obj.get('difficult', 0)})
            if mask is not None and len(mask) > 0:
                objects[-1]['mask'] = mask
            elif polygons is not None and len(polygons) > 0:
                objects[-1]['polygons'] = [np.array(p) for p in polygons]
        return objects
