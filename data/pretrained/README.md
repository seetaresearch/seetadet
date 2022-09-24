# Pretrained Models

## Introduction

This folder is kept for the pretrained models.

## ImageNet Pretrained Models

### Common settings

- ResNet models were trained with 200 epochs follow the procedure in arXiv.1812.01187.
- Channels of the first conv are inverted if trained in ``RGB`` format.

### ResNet

| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| [R50](https://dragon.seetatech.com/download/seetadet/pretrained/R-50_in1k_cls90e.pkl) | 90e | 76.53 | 93.16 | Ours |
| [R50](https://dragon.seetatech.com/download/seetadet/pretrained/R-50_in1k_cls200e.pkl) | 200e | 78.64 | 94.30 | Ours |
| [R50-A](https://dragon.seetatech.com/download/seetadet/pretrained/R-50-A_in1k_cls120e.pkl) | 120e | 75.30 | 92.20 | MSRA |

### MobileNet

| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| [MobileNetV2](https://dragon.seetatech.com/download/seetadet/pretrained/MobileNetV2_in1k_cls300e.pkl) | 300e | 71.88 | 90.29 | TorchVision |
| [MobileNetV3L](https://dragon.seetatech.com/download/seetadet/pretrained/MobileNetV3L_in1k_cls600e.pkl) | 600e | 74.04 | 91.34 | TorchVision |

### VGG

| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| [VGG16-FCN](https://dragon.seetatech.com/download/seetadet/pretrained/VGG-16-FCN_in1k.pkl) | - | - | - | weiliu89 |

