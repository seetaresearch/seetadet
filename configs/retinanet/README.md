# Focal Loss for Dense Object Detection

## Introduction
```
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```

## COCO Object Detection Baselines

| Model | Lr sched | Infer time (s/im) | box AP | Download |
| :---: | :------: | :---------------: | :----: | :------: |
| [R-50-FPN-800](coco_retinanet_R-50-FPN_800_1x.yml) | 1x | 0.051 | 37.4 | [model](https://dragon.seetatech.com/download/models/seetadet/retinanet/coco_retinanet_R-50-FPN_800_1x/model_final.pkl) |
| [R-50-FPN-800](coco_retinanet_R-50-FPN_800_2x.yml) | 2x | 0.051 | 39.1 | [model](https://dragon.seetatech.com/download/models/seetadet/retinanet/coco_retinanet_R-50-FPN_800_2x/model_final.pkl) |

## Pascal VOC Object Detection Baselines

| Model | Lr sched | Infer time (s/im) | AP@0.5 | Download |
| :---: | :------: | :---------------: | :----: | :------: |
| [R-50-FPN-512](voc_retinanet_R-50-FPN_512_120e.yml) | 120e | 0.017 | 83.0 | [model](https://dragon.seetatech.com/download/models/seetadet/retinanet/voc_retinanet_R-50-FPN_512/model_final.pkl) |
| [R-50-FPN-512](voc_retinanet_R-50-FPN_640_120e.yml) | 120e | 0.017 | 83.0 | [model](https://dragon.seetatech.com/download/models/seetadet/retinanet/voc_retinanet_R-50-FPN_512/model_final.pkl) |
