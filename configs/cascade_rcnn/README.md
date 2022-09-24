# Cascade R-CNN

## Introduction

```
@inproceedings{cai2018cascade,
  title={Cascade R-CNN: Delving into High Quality Object Detection},
  author={Cai, Zhaowei and Vasconcelos, Nuno},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6154--6162},
  year={2018}
}
```

## COCO Instance Segmentation Baselines

| Model | Lr sched | Infer time (fps) | box AP | mask AP | Download |
| :---: | :------: | :---------------: | :----: | :-----: | :------: |
| [R50-FPN](coco_cascade_rcnn_R_50_FPN_1x.yml) | 1x | 31.25 | 42.0 | 36.4 | [model](https://dragon.seetatech.com/download/seetadet/cascade_rcnn/coco_cascade_rcnn_R_50_FPN_1x/model_667640de.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/cascade_rcnn/coco_cascade_rcnn_R_50_FPN_1x/logs.json) |
| [R50-FPN](coco_cascade_rcnn_R_50_FPN_3x.yml) | 3x | 31.25 | 44.1 | 38.3 | [model](https://dragon.seetatech.com/download/seetadet/cascade_rcnn/coco_cascade_rcnn_R_50_FPN_3x/model_6df51df0.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/cascade_rcnn/coco_cascade_rcnn_R_50_FPN_3x/logs.json) |
