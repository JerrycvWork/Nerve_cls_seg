# Nerve_cls_seg

> **Authors:** 
> Alex Ngai Nick Wong, 
> Zebang He, 
> Jung Sun Yoo, 

## 1. Preface

- This repository provides code for "_**Nerve Classification and Segmentation**_"

- If you have any questions about our paper, feel free to contact me. And if you are using the work for your research, please cite this paper ([BibTeX](#4-citation)).


### 1.1. :fire: NEWS :fire:

- [2022/06/22] Release training/testing code.

- [2022/06/22] Create repository.


### 1.2. Table of Contents

- [Nerve Classification and Segmentation](#pranet--parallel-reverse-attention-network-for-polyp-segmentation--miccai-2020-)
  * [1. Preface](#1-preface)
    + [1.1. :fire: NEWS :fire:](#11--fire--news--fire-)
    + [1.2. Table of Contents](#12-table-of-contents)
    + [1.3. State-of-the-art approaches](#13-SOTAs)
  * [2. Overview](#2-overview)
    + [2.1. Introduction](#21-introduction)
    + [2.2. Framework Overview](#22-framework-overview)
    + [2.3. Qualitative Results](#23-qualitative-results)
  * [3. Proposed Baseline](#3-proposed-baseline)
    + [3.1 Training/Testing](#31-training-testing)
    + [3.2 Evaluating your trained model:](#32-evaluating-your-trained-model-)
    + [3.3 Pre-computed maps:](#33-pre-computed-maps)
  * [4. Citation](#4-citation)
  * [5. TODO LIST](#5-todo-list)
  * [6. FAQ](#6-faq)

### 1.3. State-of-the-art Approaches  




## 2. Overview


### 2.1. Introduction



### 2.2. Framework Overview



### 2.3. Qualitative Results



## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [Tensorflow]([https://github.com/pytorch/pytorch](https://www.tensorflow.org/?hl=zh-cn)) and [Keras](https://keras.io/) with 
a single GeForce RTX 3080 GPU of 10 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that PraNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n PraNet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.1

1. Downloading necessary data:


   
1. Training Configuration:


1. Testing Configuration:



### 3.2 Evaluating your trained model:



### 3.3 Pre-computed maps: 



## 4. Citation



## 5. TODO LIST



## 6. FAQ



## 7. License


