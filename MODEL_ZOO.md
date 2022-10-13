# Benchmark and Model Zoo

## Introduction

### Common settings

- All COCO models were trained on ``coco_train2017``, and evaluated on ``coco_val2017``.
- All models were trained in ``BGR`` format. For ``RGB`` pretrained, we invert channels of the first conv.
- Inference speed is measured with FPS (img/s) on a single GPU (NVIDIA RTX 3090 by default).

### Pretrained Models
Refer to [Pretrained Models](data/pretrained) for details.

## Baselines

### Faster R-CNN
Refer to [Faster R-CNN](configs/faster_rcnn) for details.

### Mask R-CNN
Refer to [Mask R-CNN](configs/mask_rcnn) for details.

### RetinaNet
Refer to [RetinaNet](configs/retinanet) for details.

### Cascade R-CNN
Refer to [Cascade R-CNN](configs/cascade_rcnn) for details.

### FCOS
Refer to [FCOS](configs/fcos) for details.

### ViTDet
Refer to [ViTDet](configs/vitdet) for details.

### Pascal VOC
Refer to [Pascal VOC](configs/pascal_voc) for details.
