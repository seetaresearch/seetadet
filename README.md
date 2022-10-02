# SeetaDet

SeetaDet is a platform implementing popular object detection algorithms.

This platform works with [**SeetaDragon**](https://dragon.seetatech.com), and uses the [**PyTorch**](https://dragon.seetatech.com/api/python/#pytorch) style.

<img src="https://dragon.seetatech.com/download/seetadet/assets/banner.png"/>

## Installation

Install from PyPI:

```bash
pip install seeta-det
```

Or, clone this repository to local disk and install:

```bash
cd seetadet && pip install .
```

You can also install from the remote repository: 

```bash
pip install git+ssh://git@github.com/seetaresearch/seetadet.git
```

If you prefer to develop locally, build but not install to ***site-packages***:

```bash
cd seetadet && python setup.py build
```

## Quick Start

### Train a detection model

```bash
cd tools
python train.py --cfg <MODEL_YAML>
```

We have provided the default YAML examples into [configs](configs).

### Test a detection model

```bash
cd tools
python test.py --cfg <MODEL_YAML> --exp_dir <EXP_DIR> --iter <ITERATION>
```

### Export a detection model to ONNX

```bash
cd tools
python export.py --cfg <MODEL_YAML> --exp_dir <EXP_DIR> --iter <ITERATION>
```

### Serve a detection model

```bash
cd tools
python serve.py --cfg <MODEL_YAML> --exp_dir <EXP_DIR> --iter <ITERATION>
```

## Benchmark and Model Zoo

Results and models are available in the [Model Zoo](MODEL_ZOO.md).

<div>
  <b>⭐️ Algorithms</b><br></br>
</div>
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/faster_rcnn">Faster R-CNN (NeurIPS'2015)</a></li>
            <li><a href="configs/pascal_voc">SSD (ECCV'2016)</a></li>
            <li><a href="configs/retinanet">RetinaNet (ICCV'2017)</a></li>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/cascade_rcnn">Cascade Mask R-CNN (CVPR'2018)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/copy_paste">CopyPaste (CVPR'2021)</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

<div>
  <b>⭐️ Architectures & Operators</b><br></br>
</div>
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li>VGG (ICLR'2015)</li>
          <li>ResNet (CVPR'2016)</li>
          <li>MobileNetV2 (CVPR'2018)</li>
          <li>MobileNetV3 (ICCV'2019)</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>FPN (CVPR'2017)</li>
            <li>BiFPN (CVPR'2020)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## License
[BSD 2-Clause license](LICENSE)
