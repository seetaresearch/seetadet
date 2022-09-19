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
"""Helper functions for polygon."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import shapely.geometry as geometry


def flip_polygons(polygons, width):
    """Flip the polygons horizontally."""
    for i, p in enumerate(polygons):
        p_flipped = p.copy()
        p_flipped[0::2] = width - p[0::2]
        polygons[i] = p_flipped
    return polygons


def crop_polygons(polygons, crop_box):
    """Crop the polygons."""
    x, y = crop_box[:2]
    crop_box = geometry.box(*crop_box).buffer(0.0)
    crop_polygons = []
    for p in polygons:
        p = p.reshape((-1, 2))
        p = geometry.Polygon(p).buffer(0.0)
        if not p.is_valid:
            continue
        cropped = p.intersection(crop_box)
        if cropped.is_empty:
            continue
        cropped = getattr(cropped, 'geoms', [cropped])
        for new_p in cropped:
            if not isinstance(new_p, geometry.Polygon) or not new_p.is_valid:
                continue
            coords = np.asarray(new_p.exterior.coords)[:-1]
            coords[:, 0] -= x
            coords[:, 1] -= y
            crop_polygons.append(coords.flatten())
    return crop_polygons
