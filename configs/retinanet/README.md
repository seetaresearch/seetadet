# RetinaNet

## Introduction

```
@inproceedings{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2980--2988},
  year={2017}
}
```

## COCO Object Detection Baselines

| Model | Lr sched | Infer time (s/im) | box AP | Download |
| :---: | :------: | :---------------: | :----: | :------: |
| [R50-FPN](coco_retinanet_R_50_FPN_1x.yml) | 1x | 38.46 | 36.6 | [model](https://dragon.seetatech.com/download/seetadet/retinanet/coco_retinanet_R_50_FPN_1x/model_d5607eb1.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/retinanet/coco_retinanet_R_50_FPN_1x/logs.json) |
| [R50-FPN](coco_retinanet_R_50_FPN_3x.yml) | 3x | 38.46 | 38.6 | [model](https://dragon.seetatech.com/download/seetadet/retinanet/coco_retinanet_R_50_FPN_3x/model_d82140f3.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/retinanet/coco_retinanet_R_50_FPN_3x/logs.json) |
