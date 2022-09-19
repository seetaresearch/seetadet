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

## License
[BSD 2-Clause license](LICENSE)
