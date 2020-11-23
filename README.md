# Vision Transformer - Pytorch
Pytorch implementation of Vision Transformer. Pretrained pytorch weights are provided which are converted from original jax/flax weights. 
This is a project of the [ASYML family](https://asyml.io/) and [CASL](https://casl-project.github.io/).


# Introduction

![Figure 1 from paper](examples/figure1.png)

Pytorch implementation of paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). 
We provide the pretrained pytorch weights which are converted from pretrained jax/flax models.
We also provide fine-tune and evaluation script. 
Similar results as in [original implementation](https://github.com/google-research/vision_transformer) are achieved.


# Installation

Create environment:
```
conda create --name vit --file requirements.txt
conda activate vit
```

# Available Models

We provide [pytorch model weights](https://drive.google.com/drive/folders/1azgrD1P413pXLJME0PjRRU-Ez-4GWN-S?usp=sharing), which are converted from original jax/flax wieghts. 
You can download them and put the files under 'weights/pytorch' to use them.

Otherwise you can download the [original jax/flax weights](https://github.com/google-research/vision_transformer) and put the fimes under 'weights/jax' to use them.
We'll convert the weights for you online.

# Datasets

Currently three datasets are supported: ImageNet2012, CIFAR10, and CIFAR100. 
To evaluate or fine-tune on these datasets, download the datasets and put them in 'data/dataset_name'. 

More datasets will be supported.


# Fine-Tune/Train
```
python src/train.py --exp-name ft --n-gpu 4 --tensorboard  --model-arch b16 --checkpoint-path weights/pytorch/imagenet21k+imagenet2012_ViT-B_16.pth --image-size 384 --batch-size 32 --data-dir data/ --dataset CIFAR10 --num-classes 10 --train-steps 10000 --lr 0.03 --wd 0.0
```


# Evaluation
Make sure you have downloaded the pretrained weights either in '.npy' format or '.pth' format
```
python src/eval.py --model-arch b16 --checkpoint-path weights/jax/imagenet21k+imagenet2012_ViT-B_16.npy --image-size 384 --batch-size 128 --data-dir data/ImageNet --dataset ImageNet --num-classes 1000
```


# Results and Models

## Pretrained Results on ImageNet2012
| upstream    | model    | dataset      | orig. jax acc  |  pytorch acc  | model link                                                                                                                                                   |
|:------------|:---------|:-------------|---------------:|--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet21k | ViT-B_16 | imagenet2012 |     84.62      |     83.90     | [checkpoint](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing) |
| imagenet21k | ViT-B_32 | imagenet2012 |     81.79      |     81.14     | [checkpoint](https://drive.google.com/file/d/1GingK9L_VcJynTCYMc3iMvCh4WG7ScBS/view?usp=sharing) |
| imagenet21k | ViT-L_16 | imagenet2012 |     85.07      |     84.94     | [checkpoint](https://drive.google.com/file/d/1YVLunKEGApaSKXZKewZz974gHt09Uwyf/view?usp=sharing) |
| imagenet21k | ViT-L_32 | imagenet2012 |     82.01      |     81.03     | [checkpoint](https://drive.google.com/file/d/1TKOa_dQaMOCL8r_rtcdB7dLGQtzBQ0ud/view?usp=sharing) |

## Fine-Tune Results on CIFAR10/100

Due to limited GPU resources, the fine-tune results are obtained by using a batch size of 32 which may impact the performance a bit.

| upstream    | model    | dataset      | orig. jax acc  |  pytorch acc  | 
|:------------|:---------|:-------------|---------------:|--------------:|
| imagenet21k | ViT-B_16 | CIFAR10      |     98.92      |     98.90     | 
| imagenet21k | ViT-B_16 | CIFAR100     |     92.26      |     91.65     | 
 

# TODO
- [ ] Colab
- [ ] Integrated into Texar


# Acknowledge
1. https://github.com/google-research/vision_transformer
2. https://github.com/lucidrains/vit-pytorch
3. https://github.com/kamalkraj/Vision-Transformer

# Contributing
Issues and Pull Requests are welcome for improving this repo. Please follow the [contribution guide](./CONTRIBUTING.md)


# License
[Apache License 2.0](./LICENSE)


# Supporting Companies and Universities
<p float="left">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/Petuum.png" width="200" align="top">
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://asyml.io/assets/institutions/cmu.png", width="200" align="top">
</p>
