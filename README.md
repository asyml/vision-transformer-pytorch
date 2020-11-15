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

Download pretrained [pytorch models](https://drive.google.com/drive/folders/1azgrD1P413pXLJME0PjRRU-Ez-4GWN-S?usp=sharing). Put the files under 'weights/pytorch'.

Download [ImageNet](http://www.image-net.org/index) dataset. Put it in 'data/ImageNet'

# Evaluation

Make sure you have downloaded the pretrained weights either in '.npy' format or '.pth' format
```
 python eval.py --model-arch b16 --checkpoint-path ../weights/jax/[model_path] --image-size 384 --batch-size 128 --data-dir ../data/ImageNet --dataset ImageNet --num-classes 1000
```


# Fine-Tune
TODO

# Results and Models

| upstream    | model    | dataset      | orig. jax acc  |  pytorch acc  | model link                                                                                                                                                   |
|:------------|:---------|:-------------|---------------:|--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet21k | ViT-B_16 | imagenet2012 |     84.62      |     83.90     | [checkpoint](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing) |
| imagenet21k | ViT-B_32 | imagenet2012 |     81.79      |     81.14     | [checkpoint](https://drive.google.com/file/d/1GingK9L_VcJynTCYMc3iMvCh4WG7ScBS/view?usp=sharing) |
| imagenet21k | ViT-L_16 | imagenet2012 |     85.07      |     84.94     | [checkpoint](https://drive.google.com/file/d/1YVLunKEGApaSKXZKewZz974gHt09Uwyf/view?usp=sharing) |
| imagenet21k | ViT-L_32 | imagenet2012 |     82.01      |     81.03     | [checkpoint](https://drive.google.com/file/d/1TKOa_dQaMOCL8r_rtcdB7dLGQtzBQ0ud/view?usp=sharing) |


# TODO List
- [ ] Fine-Tune code 
- [ ] Fine-Tune results on CIFAR10/CIFAR100
- [ ] Multi-GPU
- [ ] Integrated into Texar


# Acknowledge
1. https://github.com/google-research/vision_transformer
2. https://github.com/lucidrains/vit-pytorch
3. https://github.com/kamalkraj/Vision-Transformer
