# Sparsity by Redundancy

This repository implements several deep learning models with redundancy for image classification.

The models contains
- MLP `models/mlp.py`
- VGG `models/vgg.py`
- ResNet `models/resnet.py`

The models can be trained on CIFAR10 and CIFAR100 datasets

## A typical workaround,

Script `run.sh` illustrates a typical workaround for obtaining sparse ResNet18.

1. Train the model
```
python pretrain.py --dataset=$dataset \
                   --wd=$pretrain_wd \
                   --device=$device \
                   --output_path=$pretrain_output_path
```
2. pruning the neural network with certain threshold and then finetune the sparse network several epoches
```
python3 finetune.py --lr=1e-4 \
                    --threshold=$thr \
                    --device=$device \
                    --dataset=$dataset \
                    --resume_path $pretrain_output_path \
                    --output_path finetune/${dataset}_pretrain_wd_${pretrain_wd}/thr_${thr}
```

### Train the models on CIFAR10
![cifar10](cifar10.png)

CIFAR10 results are derived by running
```
run.sh cifar10 ${kappa} cuda:0
```
where ${kappa} is instantiated to specific parameters.

### Train the models on CIFAR100
![cifar100](cifar100.png)

CIFAR100 results are derived by running
```
run.sh cifar100 ${kappa} cuda:0
```
where ${kappa} is instantiated to specific parameters.

Plotes

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
