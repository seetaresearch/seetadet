# ViTDet

## Introduction

```
@article{li2022exploring,
  title={Exploring Plain Vision Transformer Backbones for Object Detection},
  author={Li, Yanghao and Mao, Hanzi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2203.16527},
  year={2022}
}
```

## COCO Instance Segmentation Baselines

### Common settings

- All models use Conv+SyncBN instead of Conv+LN.
- All models do not use relative position biases. (ViT-B boxAP: 51.6 -> 51.2)
- All models use bottleneck conv instead of global attention. (ViT-B boxAP: 51.2 -> 51.0)

### Mask R-CNN

| Model | Lr sched | Infer time (fps) | box AP | mask AP | Download |
| :---: | :------: | :---------------: | :----: | :-----: | :------: |
| [ViT-B](coco_mask_rcnn_vitdet_b_50e.yml) | 50e | 21.28 | 50.4 | 45.0 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_mask_rcnn_vitdet_b_50e) |
| [ViT-B](coco_mask_rcnn_vitdet_b_100e.yml) | 100e | 21.28 | 51.0 | 45.5 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_mask_rcnn_vitdet_b_100e) |
