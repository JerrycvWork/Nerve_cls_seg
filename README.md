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

In vivo SHG reflectance imaging of the sciatic nerve, the nerve-specific reflectance video is often recorded for further analysis in surgery. In clinical practice, segmenting and identifying the nerve from the nerve-specific reflectance video is of great importance since it provides valuable information for diagnosis and surgery. However, accurately segmenting out the nerve is a challenging task, for three major reasons: (i) the dataset with fine annotations is lack for deep learning-based application; (ii) the frames may not contains the nerves and only the segmentation network may hard to avoid false segmenting; (iii) nerves have a diversity of size, shape, and location.

To address these challenges, it is proposed the nerve-specific reflectance video dataset with labels of surgery stages and nerve masks and a pipeline for accurate surgery stage identification and nerve segmentation in reflectance video. Specifically, the frames of video are input to the pipeline and get the labels of the surgery stage (General Field, Tendon, and Nerve) through DenseNet201. For the frames that contain nerves, the pipeline then generates the nerve masks by the DoubleUNet segmentation network. 

Quantitative and qualitative evaluations on training, testing, and validating sets across metrics in classification and segmentation show that our pipeline improves the classification and segmentation accuracy significantly for the nerve-specific reflectance video.

### 2.3. Framework Overview

<p align="center">
    <img src="Readme_figure\pipeline.png"/> <br />
    <em> 
    Figure 1: Schematic diagram of the multi-task deep learning-based nerve imaging system to process the nerve-specific reflectance video recording.
    </em>
</p>


### 2.4. Qualitative Results

<p align="center">
    <img src="Readme_figure\quantative.png"/> <br />
    <em> 
    Figure 2: Representative video frame images of pseudo-colored nerves delineation by different neural networks. 
    </em>
</p>


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


