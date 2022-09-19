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
"""Helper functions of Non-Maximum Suppression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.ops.normalization import to_tensor
from seetadet.ops.vision import NonMaxSuppression

try:
    from seetadet.utils.nms.cython_nms import cpu_nms
    from seetadet.utils.nms.cython_nms import cpu_soft_nms
except ImportError:
    cpu_nms = cpu_soft_nms = print


def gpu_nms(dets, thresh):
    """Filter out the dets using GPU - NMS."""
    if dets.shape[0] == 0:
        return []
    scores = dets[:, 4]
    order = scores.argsort()[::-1]
    sorted_dets = to_tensor(dets[order, :])
    keep = NonMaxSuppression.apply(sorted_dets, iou_threshold=thresh)
    return order[keep.numpy()]


def nms(dets, thresh):
    """Filter out the dets using NMS."""
    if dets.shape[0] == 0:
        return []
    if cpu_nms is print:
        raise ImportError('Failed to load <cython_nms> library.')
    return cpu_nms(dets, thresh)


def soft_nms(dets, thresh, method='linear', sigma=0.5, score_thresh=0.001):
    """Filter out the dets using Soft - NMS."""
    if dets.shape[0] == 0:
        return []
    if cpu_soft_nms is print:
        raise ImportError('Failed to load <cython_nms> library.')
    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    if method not in methods:
        raise ValueError('Unknown soft nms method: ' + method)
    return cpu_soft_nms(dets, thresh, methods[method], sigma, score_thresh)
