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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.onnx.core import helper
from dragon.vm.onnx.core.exporters import utils as export_util


@export_util.register('RetinaNetDecoder')
def retinanet_decoder_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx.
    helper.add_attribute(node, 'op_type', 'RetinaNetDecoder')
    for arg in op_def.arg:
        if arg.name == 'strides':
            helper.add_attribute(node, 'strides', arg.ints)
        elif arg.name == 'ratios':
            helper.add_attribute(node, 'ratios', arg.floats)
        elif arg.name == 'scales':
            helper.add_attribute(node, 'scales', arg.floats)
        elif arg.name == 'pre_nms_topk':
            helper.add_attribute(node, 'pre_nms_topk', arg.i)
        elif arg.name == 'score_thresh':
            helper.add_attribute(node, 'score_thresh', arg.f)
    return node, const_tensors


@export_util.register('RPNDecoder')
def rpn_decoder_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx.
    helper.add_attribute(node, 'op_type', 'RPNDecoder')
    for arg in op_def.arg:
        if arg.name == 'strides':
            helper.add_attribute(node, 'strides', arg.ints)
        elif arg.name == 'ratios':
            helper.add_attribute(node, 'ratios', arg.floats)
        elif arg.name == 'scales':
            helper.add_attribute(node, 'scales', arg.floats)
        elif arg.name == 'pre_nms_topk':
            helper.add_attribute(node, 'pre_nms_topk', arg.i)
        elif arg.name == 'post_nms_topk':
            helper.add_attribute(node, 'post_nms_topk', arg.i)
        elif arg.name == 'nms_thresh':
            helper.add_attribute(node, 'nms_thresh', arg.f)
        elif arg.name == 'min_level':
            helper.add_attribute(node, 'min_level', arg.i)
        elif arg.name == 'max_level':
            helper.add_attribute(node, 'max_level', arg.i)
    return node, const_tensors
