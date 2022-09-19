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
"""Anchor generator for RPN head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class AnchorGenerator(object):
    """Generate anchors for bbox regression."""

    def __init__(self, strides, sizes, aspect_ratios,
                 scales_per_octave=1):
        self.strides = strides
        self.sizes = _align_args(strides, sizes)
        self.aspect_ratios = _align_args(strides, aspect_ratios)
        for i in range(len(self.sizes)):
            octave_sizes = []
            for j in range(1, scales_per_octave):
                scale = 2 ** (float(j) / scales_per_octave)
                octave_sizes += [x * scale for x in self.sizes[i]]
            self.sizes[i] += octave_sizes
        self.scales = [[x / y for x in z] for y, z in zip(strides, self.sizes)]
        self.cell_anchors = []
        for i in range(len(strides)):
            self.cell_anchors.append(generate_anchors(
                strides[i], self.aspect_ratios[i], self.sizes[i]))
        self.grid_shapes = None
        self.grid_anchors = None
        self.grid_coords = None

    def reset_grid(self, max_size):
        """Reset the grid."""
        self.grid_shapes = [(int(np.ceil(max_size / x)),) * 2 for x in self.strides]
        self.grid_coords = self.get_coords(self.grid_shapes)
        self.grid_anchors = self.get_anchors(self.grid_shapes)

    def num_cell_anchors(self, index=0):
        """Return number of cell anchors."""
        return self.cell_anchors[index].shape[0]

    def num_anchors(self, shapes):
        """Return the number of grid anchors."""
        return sum(self.cell_anchors[i].shape[0] * np.prod(shapes[i])
                   for i in range(len(shapes)))

    def get_slices(self, shapes):
        slices, offset = [], 0
        for i, shape in enumerate(shapes):
            num = self.cell_anchors[i].shape[0] * np.prod(shape)
            slices.append(slice(offset, offset + num))
            offset = offset + num
        return slices

    def get_coords(self, shapes):
        """Return the x-y coordinates of grid anchors."""
        xs, ys = [], []
        for i in range(len(shapes)):
            height, width = shapes[i]
            x, y = np.arange(0, width), np.arange(0, height)
            x, y = np.meshgrid(x, y)
            # Add A anchors (A,) to cell K shifts (K,)
            # to get shift coords (A, K)
            xs.append(np.tile(x.flatten(), self.cell_anchors[i].shape[0]))
            ys.append(np.tile(y.flatten(), self.cell_anchors[i].shape[0]))
        return np.concatenate(xs), np.concatenate(ys)

    def get_anchors(self, shapes):
        """Return the grid anchors."""
        grid_anchors = []
        for i in range(len(shapes)):
            h, w = shapes[i]
            shift_x = np.arange(0, w) * self.strides[i]
            shift_y = np.arange(0, h) * self.strides[i]
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()
            shifts = shifts.astype(self.cell_anchors[i].dtype)
            # Add A anchors (A, 1, 4) to cell K shifts (1, K, 4)
            # to get shift anchors (A, K, 4)
            a, k = self.num_cell_anchors(i), shifts.shape[0]
            anchors = (self.cell_anchors[i].reshape((a, 1, 4)) +
                       shifts.reshape((1, k, 4)))
            grid_anchors.append(anchors.reshape((a * k, 4)))
        return np.vstack(grid_anchors)

    def narrow_anchors(self, shapes, inds, return_anchors=False):
        """Return the valid anchors on given shapes."""
        max_shapes = self.grid_shapes
        anchors = self.grid_anchors
        x_coords, y_coords = self.grid_coords
        offset1 = offset2 = num1 = num2 = 0
        out_inds, out_anchors = [], []
        for i in range(len(max_shapes)):
            num1 += self.num_cell_anchors(i) * np.prod(max_shapes[i])
            num2 += self.num_cell_anchors(i) * np.prod(shapes[i])
            inds_keep = inds[np.where((inds >= offset1) & (inds < num1))[0]]
            anchors_keep = anchors[inds_keep] if return_anchors else None
            x, y = x_coords[inds_keep], y_coords[inds_keep]
            z = ((inds_keep - offset1) // max_shapes[i][1]) // max_shapes[i][0]
            keep = np.where((x < shapes[i][1]) & (y < shapes[i][0]))[0]
            inds_keep = (z * shapes[i][0] + y) * shapes[i][1] + x + offset2
            out_inds.append(inds_keep[keep])
            out_anchors.append(anchors_keep[keep] if return_anchors else None)
            offset1, offset2 = num1, num2
        outputs = [np.concatenate(out_inds)]
        if return_anchors:
            outputs += [np.concatenate(out_anchors)]
        return outputs[0] if len(outputs) == 1 else outputs


def generate_anchors(stride=16, ratios=(0.5, 1, 2), sizes=(32, 64, 128, 256, 512)):
    """Generate anchors by enumerating aspect ratios and sizes."""
    scales = np.array(sizes) / stride
    base_anchor = np.array([-stride / 2., -stride / 2., stride / 2., stride / 2.])
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
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
    ws = np.sqrt(w * h / ratios)
    hs = ws * ratios
    return _mkanchors(ws, hs, x_ctr, y_ctr)


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors by scales."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws, hs = w * scales, h * scales
    return _mkanchors(ws, hs, x_ctr, y_ctr)


def _align_args(strides, args):
    """Align the args to the strides."""
    args = (args * len(strides)) if len(args) == 1 else args
    assert len(args) == len(strides)
    return [[x] if not isinstance(x, (tuple, list)) else x[:] for x in args]
