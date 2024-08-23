## Attention Mono-depth: attention-enhanced transformer for monocular depth estimation of volatile kiln burden surface

## Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)

## Installation

The code is tested with Python 3.7, PyTorch 1.8.1
```
conda create -n attentiondepth python=3.7
conda activate attentiondepth
pip install -r requirements.txt
```

## Datasets
You can download our dataset from [here](https://pan.quark.cn/s/2f20eff3cea3), and then modify the data path in the config files to your dataset locations.

## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training model:
```
python train.py args_train_burden.txt
```

## Evaluation

Evaluate model:
```
python evaluate.py args_test_burden.txt
```
