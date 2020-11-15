# Vision Transformer - Pytorch Implementation and Pretrained Models
Implementation of Vision Transformer. Pretrained pytorch weights are provided converted from original jax weights. 


# Introduction

![Figure 1 from paper](examples/figure1.png)

Pytorch implementation of paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). 
We provide the pretrained pytorch weights which are converted from pretrained jax models.
We also provide fine-tune and evaluation script. 
Similar results as in [original Jax implementation](https://github.com/google-research/vision_transformer) are achieved.


# Installation

Create environment:
```
conda create --name vit --file requirements.txt
```

Download pretrained jax models from the [original jax implementation repo](https://github.com/google-research/vision_transformer). 
Put the files under 'weights/jax'.

Download pretrained pytorch models from ++. Put the files under 'weights/pytorch'.

Download ImageNet dataset from and process it according to. Put it in 'data/ImageNet'

# Evaluation

Make sure you have downloaded the pretrained weights either in '.npy' format or '.pth' format
```
 python eval.py --model-arch b16 --checkpoint-path ../weights/jax/[model_path] --image-size 384 --batch-size 128 --data-dir ../data/ImageNet --dataset ImageNet --num-classes 1000
```


# Fine-Tune
TODO

# Results and Models

| upstream    | model    | dataset      |   jax acc  |   acc   | model link                                                                                                                                                   |
|:------------|:---------|:-------------|-----------:|--------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet21k | ViT-B_16 | imagenet2012 |     0.8462 | 0.8390  | [checkpoint] |
| imagenet21k | ViT-B_32 | imagenet2012 |     0.8179 | 0.8114  | [checkpoint] |
| imagenet21k | ViT-L_16 | imagenet2012 |     0.8507 |         | [checkpoint] |
| imagenet21k | ViT-L_32 | imagenet2012 |     0.8201 |         | [checkpoint] |


# TODO List
- [] Fine-Tune code 
- [] Fine-Tune results on CIFAR10/CIFAR100
- [] Multi-GPU

# Acknowledge
1. https://github.com/google-research/vision_transformer
2. https://github.com/lucidrains/vit-pytorch
3. https://github.com/kamalkraj/Vision-Transformer
