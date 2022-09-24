# Pascal VOC

## Introduction

```
@article{everingham2010pascal,
  title={The PASCAL Visual Object Classes (VOC) Challenge},
  author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
  journal={International Journal of Computer Vision},
  volume={88},
  number={2},
  pages={303--338},
  year={2010},
  publisher={Springer}
}
```

## Object Detection Baselines

### Faster R-CNN

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :------: |
| [R50-FPN](voc_faster_rcnn_R_50_FPN_15e.yml) | 15e | 58.82 | 82.0 | [model](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_faster_rcnn_R_50_FPN_15e/model_3dcb03f9.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_faster_rcnn_R_50_FPN_15e/logs.json) |

### RetinaNet

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :------: |
| [R50-FPN](voc_retinanet_R_50_FPN_120e.yml) | 120e | 83.33 | 82.5 | [model](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_retinanet_R_50_FPN_120e/model_d2b6ebb6.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_retinanet_R_50_FPN_120e/logs.json) |

### SSD

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :--------------: | :----: | :------: |
| [VGG16-SSD300](voc_ssd300_VGG_16_120e.yml) | 120e | 166.67 | 77.8 | [model](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_ssd300_VGG_16_120e/model_3417d961.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/pascal_voc/voc_ssd300_VGG_16_120e/logs.json) |
