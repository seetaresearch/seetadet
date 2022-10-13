# FCOS

## Introduction

```
@inproceedings{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9627--9636},
  year={2019}
}
```

## COCO Object Detection Baselines

### Common settings

- All models use center sampling and normalize targets by stride.
- All models do not use learnable scale layer for regression logits.

| Model | Lr sched | Infer time (fps) | box AP | Download |
| :---: | :------: | :---------------: | :----: | :------: |
| [R50-FPN](coco_fcos_R_50_FPN_1x.yml) | 1x | 47.62 | 38.1 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_fcos_R_50_FPN_1x) |
| [R50-FPN](coco_fcos_R_50_FPN_2x.yml) | 2x | 47.62 | 40.9 | [model](https://www.codewithgpu.com/m/seetaresearch/seetadet/coco_fcos_R_50_FPN_2x) |
