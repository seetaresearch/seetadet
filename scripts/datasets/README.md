# Prepare Datasets

## Create Datasets for PASCAL VOC

We assume that raw dataset has the following structure:

```
VOC<year>
|_ JPEGImages
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ Annotations
|  |_ <im-1-name>.xml
|  |_ ...
|  |_ <im-N-name>.xml
|_ ImageSets
|  |_ Main
|  |  |_ trainval.txt
|  |  |_ test.txt
|  |  |_ ...
```

Create record and json dataset by:

```
python pascal_voc.py \
  --rec /path/to/datasets/voc_trainval0712 \
  --gt /path/to/datasets/voc_trainval0712.json \
  --images /path/to/VOC2007/JPEGImages \
           /path/to/VOC2012/JPEGImages \
  --annotations /path/to/VOC2007/Annotations \
                /path/to/VOC2012/Annotations \
  --splits /path/to/VOC2007/ImageSets/Main/trainval.txt \
           /path/to/VOC2012/ImageSets/Main/trainval.txt
```

## Create Datasets for COCO

We assume that raw dataset has the following structure:

```
COCO
|_ images
|  |_ train2017
|  |  |_ <im-1-name>.jpg
|  |  |_ ...
|  |  |_ <im-N-name>.jpg
|_ annotations
|  |_ instances_train2017.json
|  |_ ...
```

Create record dataset by:

```
python coco.py \
  --rec /path/to/datasets/coco_train2017 \
  --images /path/to/COCO/images/train2017 \
  --annotations /path/to/COCO/annotations/instances_train2017.json
```
