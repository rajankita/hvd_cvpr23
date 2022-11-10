# Learning from Infants: Upside of Critical Learning Periods in Deep Networks
 
Our implementation is based on these repositories:
- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)
- [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)



## Updates
**23 November, 2021**: First version

## Getting Started
### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy
- pytorch-gradcam (https://pypi.org/project/pytorch-gradcam/ )
- matplotlib

## Evaluation script

To evaluate trained models on clean and corrupted dataset, run ```test_corruption.py```. It takes the following arguments:

 - ``` --net_type``` model architecture. Supported values are ```allconv``` and ```resnet```
  - ```--depth``` network depth, if using resnet
  - ```--dataset``` the training dataset. Supported values are ```cifar10```, ```tinyimagenet``` and ```imagenet```
  - ```--pretrained``` path to pretrained model weights
  - ```--results``` output directory


## Training Script

To train a network, use the ```train_clewr.py``` script. It takes the following arguments:
  
  - ``` --net_type``` model architecture. Supported values are ```allconv``` and ```resnet```
  - ```--depth``` network depth, if using resnet
  - ```--dataset``` the training dataset. Supported values are ```cifar10```, ```tinyimagenet``` and ```imagenet```
  - ```--expname``` experiment name. Trained models are saved in a directory of the same name
  - ```--epochs``` training epochs per segment of the curriculum
  - ```--sigmas``` sigma values per segment of the curriculum
  - ```--batch_size``` training batch size
  - ```--lr``` initial learning rate

Example usage of training script for training an All-Convolutional network on CIFAR-10 dataset:

### Vanilla training
```
python train_clewr.py --net_type allconv --dataset cifar10 --batch_size 128 --lr 0.1 --expname runs/cifar10/allconv/vanilla --epochs 120 --sigmas 0
```

### CLEWR training
```
python train_clewr.py --net_type allconv --dataset cifar10 --batch_size 128 --lr 0.1 --expname runs/cifar10/allconv/clewr --epochs 8 16 96 --sigmas 2 1 0
```

### Imagenet training
For training on imagenet, use the following script. We train using two GPUs on imagenet. 
```
CUDA_VISIBLE_DEVICES=0,1 python train_clewr_imagenet.py -b 2048 <path_to_imagenet_dataset>
```

## Additional scripts

### [AugMix](https://github.com/google-research/augmix)
Train AugMix on CIFAR-10 
```
python augmix_cifar.py -m allconv -wd 0.0001 --epochs 120 --no-jsd 
```

### CLEWR+AugMix
Train CLEWR+AugMix on CIFAR-10
```
python train_clewr+augmix.py -m allconv -wd 0.0001 --no-jsd --epochs 8 16 94 --sigmas 2 1 0 -s runs/cifar10/allconv/clewr_exp+augmix/
```

### Blur-Augment
Train with augmentation with blurred images on CIFAR-10
```
python train_blur_augment.py --net_type allconv --dataset cifar10 --batch_size 128 --lr 0.1 --expname runs/cifar10/allconv/blur_aug --epochs 120 --sigma_max 2
```
### CLEWR without replay
Train CLEWR without replay of previous blur kernels
```
python train_clewr_sans_replay.py --net_type allconv --dataset cifar10 --batch_size 128 --lr 0.1 --expname runs/cifar10/allconv/clewr_exp_sans_replay --epochs 8 16 96 --sigmas 2 1 0 
```