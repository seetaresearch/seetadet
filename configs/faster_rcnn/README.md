# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Introduction
```
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```

## COCO Object Detection Baselines

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :-----: |
| [R50-FPN](coco_faster_rcnn_R_50_FPN_1x.yml) | 1x | 37.04 | 37.7 | [model](https://dragon.seetatech.com/download/seetadet/faster_rcnn/coco_faster_rcnn_R_50_FPN_1x/model_7abb52ab.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/faster_rcnn/coco_faster_rcnn_R_50_FPN_1x/logs.json) |
| [R50-FPN](coco_faster_rcnn_R_50_FPN_3x.yml) | 3x | 37.04 | 39.8 | [model](https://dragon.seetatech.com/download/seetadet/faster_rcnn/coco_faster_rcnn_R_50_FPN_3x/model_04e548ca.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/faster_rcnn/coco_faster_rcnn_R_50_FPN_3x/logs.json) |
