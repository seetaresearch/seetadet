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
"""Default configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seetadet.core.config.yacs import CfgNode

_C = cfg = CfgNode()

# ------------------------------------------------------------
# Training options
# ------------------------------------------------------------
_C.TRAIN = CfgNode()

# Initialize network with weights from this file
_C.TRAIN.WEIGHTS = ''

# The train dataset
_C.TRAIN.DATASET = ''

# The loader type for training
_C.TRAIN.LOADER = 'det_train'

# The number of workers to load train data
_C.TRAIN.NUM_WORKERS = 3

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image shortest side
_C.TRAIN.SCALES = (640,)

# Range to jitter the image scales randomly
_C.TRAIN.SCALES_RANGE = (1.0, 1.0)

# Longest side to resize the input image
_C.TRAIN.MAX_SIZE = 1000

# Size to crop the input image
_C.TRAIN.CROP_SIZE = 0

# Images to use per mini-batch
_C.TRAIN.IMS_PER_BATCH = 1

# Use the difficult (occluded/crowd) objects
_C.TRAIN.USE_DIFF = False

# The probability to distort the color
_C.TRAIN.COLOR_JITTER = 0.0

# ------------------------------------------------------------
# Testing options
# ------------------------------------------------------------
_C.TEST = CfgNode()

# The test dataset
_C.TEST.DATASET = ''

# THE JSON format dataset with annotations for evaluation
_C.TEST.JSON_DATASET = ''

# The loader type for testing
_C.TEST.LOADER = 'det_test'

# The evaluator type for dataset
_C.TEST.EVALUATOR = ''

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
_C.TEST.SCALES = (640,)

# Max pixel size of the longest side of a scaled input image
_C.TEST.MAX_SIZE = 1000

# Size to crop the input image
_C.TEST.CROP_SIZE = 0

# Images to use per mini-batch
_C.TEST.IMS_PER_BATCH = 1

# The threshold for predicting boxes
_C.TEST.SCORE_THRESH = 0.05

# Overlap threshold used for NMS
_C.TEST.NMS_THRESH = 0.5

# Maximum number of detections to return per image
# 100 is based on the limit established for the COCO dataset
_C.TEST.DETECTIONS_PER_IM = 100

# ------------------------------------------------------------
# Model options
# ------------------------------------------------------------
_C.MODEL = CfgNode()

# The model type
_C.MODEL.TYPE = ''

# The compute precision
_C.MODEL.PRECISION = 'float32'

# The name for each object class
_C.MODEL.CLASSES = ['__background__']

# Pixel mean and stddev values for image normalization (BGR order)
_C.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_STD = [57.375, 57.12, 58.395]

# Focal loss parameters
_C.MODEL.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.FOCAL_LOSS_GAMMA = 2.0

# ------------------------------------------------------------
# Backbone options
# ------------------------------------------------------------
_C.BACKBONE = CfgNode()

# The backbone type
_C.BACKBONE.TYPE = ''

# The normalization in backbone modules
_C.BACKBONE.NORM = 'FrozenBN'

# The drop path rate in backbone
_C.BACKBONE.DROP_PATH_RATE = 0.0

# Freeze the first stages/blocks of backbone
_C.BACKBONE.FREEZE_AT = 2

# Stride of the coarsest feature
# This is needed so the input can be padded properly
_C.BACKBONE.COARSEST_STRIDE = 32

# ------------------------------------------------------------
# FPN options
# ------------------------------------------------------------
_C.FPN = CfgNode()

# Finest level of the FPN pyramid
_C.FPN.MIN_LEVEL = 3

# Coarsest level of the FPN pyramid
_C.FPN.MAX_LEVEL = 7

# Starting level of the top-down fusing
_C.FPN.FUSE_LEVEL = 5

# Number of blocks to stack in the FPN
_C.FPN.NUM_BLOCKS = 1

# Channel dimension of the FPN feature levels
_C.FPN.DIM = 256

# The FPN conv module
_C.FPN.CONV = 'Conv2d'

# The fpn normalization module
_C.FPN.NORM = ''

# The fpn activation module
_C.FPN.ACTIVATION = ''

# The feature fusion method
_C.FPN.FUSE_TYPE = 'sum'

# ------------------------------------------------------------
# Anchor generator options
# ------------------------------------------------------------

_C.ANCHOR_GENERATOR = CfgNode()

# The stride of each level
_C.ANCHOR_GENERATOR.STRIDES = [8, 16, 32, 64, 128]

# The anchor size of each stride
_C.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]

# The aspect ratios of each stride
_C.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# ------------------------------------------------------------
# RPN options
# ------------------------------------------------------------
_C.RPN = CfgNode()

# Total number of rpn training anchors per image
_C.RPN.BATCH_SIZE = 256

# Fraction of foreground anchors per training batch
_C.RPN.POSITIVE_FRACTION = 0.5

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.RPN.POSITIVE_OVERLAP = 0.7

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.RPN.NEGATIVE_OVERLAP = 0.3

# NMS threshold used on RPN proposals
_C.RPN.NMS_THRESH = 0.7

# Number of top scoring boxes to keep before NMS to RPN proposals
_C.RPN.PRE_NMS_TOPK_TRAIN = 2000
_C.RPN.PRE_NMS_TOPK_TEST = 1000

# Number of top scoring boxes to keep after NMS to RPN proposals
_C.RPN.POST_NMS_TOPK_TRAIN = 1000
_C.RPN.POST_NMS_TOPK_TEST = 1000

# The number of conv layers to stack in the head
_C.RPN.NUM_CONV = 1

# The optional loss for bbox regression
_C.RPN.BBOX_REG_LOSS_TYPE = 'l1'

# Weight for bbox regression loss
_C.RPN.BBOX_REG_LOSS_WEIGHT = 1.0

# ------------------------------------------------------------
# RetinaNet options
# ------------------------------------------------------------
_C.RETINANET = CfgNode()

# Number of conv layers to stack in the head
_C.RETINANET.NUM_CONV = 4

# The head conv module
_C.RETINANET.CONV = 'Conv2d'

# The head normalization module
_C.RETINANET.NORM = ''

# The head activation module
_C.RETINANET.ACTIVATION = 'ReLU'

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.RETINANET.POSITIVE_OVERLAP = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.RETINANET.NEGATIVE_OVERLAP = 0.4

# Number of top scoring boxes to keep before NMS
_C.RETINANET.PRE_NMS_TOPK = 1000

# The bbox regression loss type
_C.RETINANET.BBOX_REG_LOSS_TYPE = 'l1'

# The weight for bbox regression loss
_C.RETINANET.BBOX_REG_LOSS_WEIGHT = 1.0

# ------------------------------------------------------------
# FastRCNN options
# ------------------------------------------------------------
_C.FAST_RCNN = CfgNode()

# Total number of training RoIs per image
_C.FAST_RCNN.BATCH_SIZE = 512

# The finest level of RoI feature
_C.FAST_RCNN.MIN_LEVEL = 2

# The coarsest level of RoI feature
_C.FAST_RCNN.MAX_LEVEL = 5

# Fraction of foreground RoIs per training batch
_C.FAST_RCNN.POSITIVE_FRACTION = 0.25

# IoU overlap ratio for labeling a RoI as positive
# RoIs with >= iou overlap are labeled positive
_C.FAST_RCNN.POSITIVE_OVERLAP = 0.5

# IoU overlap ratio for labeling a RoI as negative
# RoIs with < iou overlap are labeled negative
_C.FAST_RCNN.NEGATIVE_OVERLAP = 0.5

# RoI pooler type
_C.FAST_RCNN.POOLER_TYPE = 'RoIAlignV2'

# The output size of of RoI pooler
_C.FAST_RCNN.POOLER_RESOLUTION = 7

# The resampling window size of RoI pooler
_C.FAST_RCNN.POOLER_SAMPLING_RATIO = 0

# The number of conv layers to stack in the head
_C.FAST_RCNN.NUM_CONV = 0

# The number of fc layers to stack in the head
_C.FAST_RCNN.NUM_FC = 2

# The hidden dimension of conv head
_C.FAST_RCNN.CONV_HEAD_DIM = 256

# The hidden dimension of fc head
_C.FAST_RCNN.FC_HEAD_DIM = 1024

# The head normalization module
_C.FAST_RCNN.NORM = ''

# Use class agnostic for bbox regression or not
_C.FAST_RCNN.BBOX_REG_CLS_AGNOSTIC = False

# The bbox regression loss type
_C.FAST_RCNN.BBOX_REG_LOSS_TYPE = 'l1'

# The weight for bbox regression loss
_C.FAST_RCNN.BBOX_REG_LOSS_WEIGHT = 1.0

# The weights on (dx, dy, dw, dh) for normalizing bbox regression targets
_C.FAST_RCNN.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# ------------------------------------------------------------
# MaskRCNN options
# ------------------------------------------------------------
_C.MASK_RCNN = CfgNode()

# RoI pooler type
_C.MASK_RCNN.POOLER_TYPE = 'RoIAlignV2'

# The output size of of RoI pooler
_C.MASK_RCNN.POOLER_RESOLUTION = 14

# The resampling window size of RoI pooler
_C.MASK_RCNN.POOLER_SAMPLING_RATIO = 0

# The number of conv layers to stack in the head
_C.MASK_RCNN.NUM_CONV = 4

# The hidden dimension of conv head
_C.MASK_RCNN.CONV_HEAD_DIM = 256

# The head normalization module
_C.MASK_RCNN.NORM = ''

# ------------------------------------------------------------
# CascadeRCNN options
# ------------------------------------------------------------
_C.CASCADE_RCNN = CfgNode()

# Make mask predictions or not
_C.CASCADE_RCNN.MASK_ON = False

# IoU overlap ratios for labeling a RoI as positive
# RoIs with >= iou overlap are labeled positive
_C.CASCADE_RCNN.POSITIVE_OVERLAP = (0.5, 0.6, 0.7)

# The weights on (dx, dy, dw, dh) for normalizing bbox regression targets
_C.CASCADE_RCNN.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)

# ------------------------------------------------------------
# SSD options
# ------------------------------------------------------------
_C.SSD = CfgNode()

# Fraction of foreground anchors per training batch
_C.SSD.POSITIVE_FRACTION = 0.25

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.SSD.POSITIVE_OVERLAP = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.SSD.NEGATIVE_OVERLAP = 0.5

# Number of top scoring boxes to keep before NMS
_C.SSD.PRE_NMS_TOPK = 300

# The optional loss for bbox regression
# Values supported: 'l1', 'smooth_l1', 'giou'
_C.SSD.BBOX_REG_LOSS_TYPE = 'l1'

# Weight for bbox regression loss
_C.SSD.BBOX_REG_LOSS_WEIGHT = 1.0

# The weights on (dx, dy, dw, dh) for normalizing bbox regression targets
_C.SSD.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# ------------------------------------------------------------
# Solver options
# ------------------------------------------------------------
_C.SOLVER = CfgNode()

# The interval to display logs
_C.SOLVER.DISPLAY = 20

# The interval to snapshot a model
_C.SOLVER.SNAPSHOT_EVERY = 5000

# Prefix to yield the path: <prefix>_iter_XYZ.pkl
_C.SOLVER.SNAPSHOT_PREFIX = ''

# Loss scaling factor for mixed precision training
_C.SOLVER.LOSS_SCALE = 1024.0

# Maximum number of SGD iterations
_C.SOLVER.MAX_STEPS = 40000

# Base learning rate for the specified scheduler
_C.SOLVER.BASE_LR = 0.001

# Minimal learning rate for the specified scheduler
_C.SOLVER.MIN_LR = 0.0

# The decay intervals for LRScheduler
_C.SOLVER.DECAY_STEPS = []

# The decay factor for exponential LRScheduler
_C.SOLVER.DECAY_GAMMA = 0.1

# Warm up to ``BASE_LR`` over this number of steps
_C.SOLVER.WARM_UP_STEPS = 1000

# Start the warm up from ``BASE_LR`` * ``FACTOR``
_C.SOLVER.WARM_UP_FACTOR = 1.0 / 1000

# The type of optimizier
_C.SOLVER.OPTIMIZER = 'SGD'

# The type of lr scheduler
_C.SOLVER.LR_POLICY = 'steps_with_decay'

# The layer-wise lr decay
_C.SOLVER.LAYER_LR_DECAY = 1.0

# Momentum to use with SGD
_C.SOLVER.MOMENTUM = 0.9

# L2 regularization for weight parameters
_C.SOLVER.WEIGHT_DECAY = 0.0001

# L2 norm factor for clipping gradients
_C.SOLVER.CLIP_NORM = 0.0

# ------------------------------------------------------------
# Misc options
# ------------------------------------------------------------
# Number of GPUs for distributed training
_C.NUM_GPUS = 1

# Random seed for reproducibility
_C.RNG_SEED = 3

# Default GPU device index
_C.GPU_ID = 0
