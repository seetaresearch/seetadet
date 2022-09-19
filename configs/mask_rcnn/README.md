# Mask R-CNN

## Introduction
```
@article{He_2017,
   title={Mask R-CNN},
   journal={2017 IEEE International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
   year={2017},
   month={Oct}
}
```

## COCO Instance Segmentation Baselines

| Model | Lr sched | Infer time (fps) | box AP | mask AP | Download |
| :---: | :------: | :---------------: | :----: | :-----: | :------: |
| [R50-FPN](coco_mask_rcnn_R_50_FPN_1x.yml) | 1x | 30.30 | 38.3 | 34.9 | [model](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_1x/model_b27317db.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_1x/logs.json) |
| [R50-FPN](coco_mask_rcnn_R_50_FPN_3x.yml) | 3x | 30.30 | 40.7 | 36.8 | [model](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_3x/model_6f7e3878.pkl) &#124; [log](https://dragon.seetatech.com/download/seetadet/mask_rcnn/coco_mask_rcnn_R_50_FPN_3x/logs.json) |
