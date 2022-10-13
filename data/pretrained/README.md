# Pretrained Models

## Introduction

This folder is kept for the pretrained models. Models are available at [CodeWithGPU](https://www.codewithgpu.com/m/seetaresearch/seetadet/pretrained).

## ImageNet Pretrained Models

### Common settings
- ResNet models were trained with 200 epochs follow the procedure in arXiv.1812.01187.
- Channels of the first conv are inverted if trained in ``RGB`` format.

### ResNet
| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| R50-A | 120e | 75.30 | 92.20 | KaimingHe/deep-residual-networks |
| R50 | 90e | 76.53 | 93.16 | Ours |
| R50 | 200e | 78.64 | 94.30 | Ours |

### VisionTransformer
| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| ViT-B | mae1600e | 83.6 | - | facebookresearch/mae |

### MobileNet
| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| MobileNetV2 | 300e | 71.88 | 90.29 | pytorch/vision |
| MobileNetV3L | 600e | 74.04 | 91.34 | pytorch/vision |

### VGG
| Model | Lr sched | Acc@1 | Acc@5 | Source |
| :---: | :------: | :---: | :---: | :----: |
| VGG16-FCN | - | - | - | weiliu89/caffe |

