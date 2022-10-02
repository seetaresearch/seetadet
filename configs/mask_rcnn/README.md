# Mask R-CNN

## Introduction

```
@inproceedings{he2017mask,
  title={Mask R-CNN},
  author={He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2961--2969},
  year={2017}
}
```

## COCO Instance Segmentation Baselines

| Model | Lr sched | Infer time (fps) | box AP | mask AP | Download |
| :---: | :------: | :---------------: | :----: | :-----: | :------: |
| [R50-FPN](coco_mask_rcnn_R_50_FPN_1x.yml) | 1x | 38.46 | 38.3 | 35.1 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_mask_rcnn_R_50_FPN_1x) |
| [R50-FPN](coco_mask_rcnn_R_50_FPN_3x.yml) | 3x | 38.46 | 40.7 | 36.9 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_mask_rcnn_R_50_FPN_3x) |
