# Nerve_cls_seg

> **Authors:** 
> Alex Ngai Nick Wong, 
> Zebang He, 
> Jung Sun Yoo, 

## 1. Preface

- This repository provides code for "_**Nerve Classification and Segmentation**_"

- If you have any questions about our paper, feel free to contact us. And if you are using the work for your research, please cite this paper ([BibTeX](#4-citation)).


### 1.1. :fire: NEWS :fire:

- [2022/06/22] Release training/testing code.

- [2022/06/22] Create repository.


### 1.2. Table of Contents

- Nerve Classification and Segmentation
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

1. Huang, Gao et al. “Densely Connected Convolutional Networks.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 2261-2269.

   paper link: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099726

2. Jha, Debesh et al. “DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation.” 2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS) (2020): 558-564.

   paper link: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9183321


## 2. Overview




### 2.1. Introduction



### 2.2. Framework Overview



### 2.3. Qualitative Results



## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [Tensorflow](https://www.tensorflow.org/?hl=zh-cn) and [Keras](https://keras.io/) with 
a single GeForce RTX 3080 GPU of 10 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that the pipeline is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n Nerveseg python=3.7`.
    
    + Installing necessary packages: Tensorflow 1.13.1, Keras 2.3.0

    > Note that, the Official Source of Tensorflow 1.13.1 may not support the Ampere Structure GPU (NVIDIA RTX 30 Series), and you may need to download Tensorflow at the [source of NVIDIA](https://github.com/NVIDIA/tensorflow).

1. Downloading necessary data:


   
1. Training Configuration:


1. Testing Configuration:



### 3.2 Evaluating your trained model:



### 3.3 Pre-computed maps: 



## 4. Citation



## 5. TODO LIST



## 6. FAQ



## 7. License


