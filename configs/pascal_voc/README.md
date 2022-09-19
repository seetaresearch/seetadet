# Pascal VOC

## Introduction
```latex
@Article{Everingham10,
   author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
   title = "The Pascal Visual Object Classes (VOC) Challenge",
   journal = "International Journal of Computer Vision",
   volume = "88",
   year = "2010",
   number = "2",
   month = jun,
   pages = "303--338",
}
```

## Object Detection Baselines

### Faster R-CNN

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :------: |
| [R50-FPN](voc_faster_rcnn_R_50_FPN_15e.yml) | 15e | 47.62 | 82.1 | [model](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_faster_rcnn_R_50_FPN_15e/model_3dcb03f9.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_faster_rcnn_R_50_FPN_15e/logs.json) |

### RetinaNet

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :------: |
| [R50-FPN](voc_retinanet_R_50_FPN_120e.yml) | 120 | 58.82 | 82.4 | [model](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_retinanet_R_50_FPN_120e/model_1ae4cd3d.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_retinanet_R_50_FPN_120e/logs.json) |

### SSD

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :------: |
| [VGG16-SSD300](voc_ssd300_VGG_16_120e.yml) | 120 | 125 | 77.8 | [model](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_ssd300_VGG_16_120e/model_3417d961.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_ssd300_VGG_16_120e/logs.json) |
