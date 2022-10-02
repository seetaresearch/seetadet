# Faster R-CNN

## Introduction

```
@article{ren2015faster,
  title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal={Advances in Neural Information Processing Systems},
  volume={28},
  year={2015}
}
```

## COCO Object Detection Baselines

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :-----: |
| [R50-FPN](coco_faster_rcnn_R_50_FPN_1x.yml) | 1x | 45.45 | 37.4 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_faster_rcnn_R_50_FPN_1x) |
| [R50-FPN](coco_faster_rcnn_R_50_FPN_3x.yml) | 3x | 45.45 | 40.0 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_faster_rcnn_R_50_FPN_3x) |
