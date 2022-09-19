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
"""Anchor generator for SSD head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class AnchorGenerator(object):
    """Generate anchors for bbox regression."""

    def __init__(self, strides, sizes, aspect_ratios):
        self.strides = strides
        self.sizes = _align_args(strides, sizes)
        self.aspect_ratios = _align_args(strides, aspect_ratios)
        self.scales = [[x / y for x in z] for y, z in zip(strides, self.sizes)]
        self.cell_anchors = []
        for i in range(len(strides)):
            self.cell_anchors.append(generate_anchors(
                self.aspect_ratios[i], self.sizes[i]))
        self.grid_shapes = None
        self.grid_anchors = None

    def reset_grid(self, max_size):
        """Reset the grid."""
        self.grid_shapes = [(int(np.ceil(max_size / x)),) * 2 for x in self.strides]
        self.grid_anchors = self.get_anchors(self.grid_shapes)

    def num_cell_anchors(self, index=0):
        """Return number of cell anchors."""
        return self.cell_anchors[index].shape[0]

    def get_anchors(self, shapes):
        """Return the grid anchors."""
        grid_anchors = []
        for i in range(len(shapes)):
            h, w = shapes[i]
            shift_x = (np.arange(0, w) + 0.5) * self.strides[i]
            shift_y = (np.arange(0, h) + 0.5) * self.strides[i]
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()
            shifts = shifts.astype(self.cell_anchors[i].dtype)
            # Add a anchors (1, A, 4) to cell K shifts (K, 1, 4)
            # to get shift anchors (K, A, 4) and reshape to (K * A, 4)
            a = self.cell_anchors[i].shape[0]
            k = shifts.shape[0]
            anchors = (self.cell_anchors[i].reshape((1, a, 4)) +
                       shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
            grid_anchors.append(anchors.reshape((k * a, 4)))
        return np.vstack(grid_anchors)


def generate_anchors(ratios, sizes):
    """Generate anchors by enumerating aspect ratios and sizes."""
    min_size, max_size = sizes
    base_anchor = np.array([-min_size / 2., -min_size / 2.,
                            min_size / 2., min_size / 2.])
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    size_anchors = _size_enum(base_anchor, min_size, max_size)
    anchors = np.vstack([ratio_anchors[:1], size_anchors, ratio_anchors[1:]])
    return anchors.astype('float32')


def _whctrs(anchor):
    """Return the xywh of an anchor."""
    w = anchor[2] - anchor[0]
    h = anchor[3] - anchor[1]
    x_ctr = anchor[0] + 0.5 * w
    y_ctr = anchor[1] + 0.5 * h
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Return a sef of anchors by widths, heights and center."""
    ws, hs = ws[:, np.newaxis], hs[:, np.newaxis]
    return np.hstack((x_ctr - 0.5 * ws, y_ctr - 0.5 * hs,
                      x_ctr + 0.5 * ws, y_ctr + 0.5 * hs))


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors by aspect ratios."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    hs = np.sqrt(w * h / ratios)
    ws = hs * ratios
    return _mkanchors(ws, hs, x_ctr, y_ctr)


def _size_enum(anchor, min_size, max_size):
    """Enumerate a anchor for size wrt base_anchor."""
    _, _, x_ctr, y_ctr = _whctrs(anchor)
    ws = hs = np.sqrt([min_size * max_size])
    return _mkanchors(ws, hs, x_ctr, y_ctr)


def _align_args(strides, args):
    """Align the args to the strides."""
    args = (args * len(strides)) if len(args) == 1 else args
    assert len(args) == len(strides)
    return [[x] if not isinstance(x, (tuple, list)) else x[:] for x in args]


if __name__ == '__main__':
    anchor_generator = AnchorGenerator(
        strides=(8, 16, 32, 64, 100, 300),
        sizes=((30, 60), (60, 110), (110, 162),
               (162, 213), (213, 264), (264, 315)),
        aspect_ratios=((1, 2, 0.5),
                       (1, 2, 0.5, 3, 0.33),
                       (1, 2, 0.5, 3, 0.33),
                       (1, 2, 0.5, 3, 0.33),
                       (1, 2, 0.5),
                       (1, 2, 0.5)))
    anchor_generator.reset_grid(max_size=300)
    assert anchor_generator.grid_anchors.shape == (8732, 4)
