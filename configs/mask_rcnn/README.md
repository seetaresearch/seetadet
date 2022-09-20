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
| [R50-FPN](coco_mask_rcnn_R_50_FPN_1x.yml) | 1x | 30.30 | 38.3 | 34.9 | [model](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_1x/model_b27317db.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_1x/logs.json) |
| [R50-FPN](coco_mask_rcnn_R_50_FPN_3x.yml) | 3x | 30.30 | 40.7 | 36.8 | [model](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_3x/model_6f7e3878.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_3x/logs.json) |
