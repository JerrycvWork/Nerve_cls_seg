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
    + [3.3 More Result](#33-More-Result) 
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

> Evaluation of Classification
<p align="center">
    <img src="Readme_figure\cls_performance.png"/> <br />
    <em> 
    Figure 2: The performance of different classification models for “nerve” category images recognition. 
    </em>
</p>

> Evaluation of Segmentation
<p align="center">
    <img src="Readme_figure\seg_performance.png"/> <br />
    <em> 
    Figure 3: The performance of different neural network models for nerve delineation. 
    </em>
</p>

> Visualization of Segmentation
<p align="center">
    <img src="Readme_figure\quantative.png"/> <br />
    <em> 
    Figure 4: Representative video frame images of pseudo-colored nerves delineation by different neural networks. 
    </em>
</p>


## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [Tensorflow](https://www.tensorflow.org/?hl=zh-cn) and [Keras](https://keras.io/) with 
a single GeForce RTX 3080 GPU of 10 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


**1. Configuring your environment (Prerequisites):** 
   
  > Note that the pipeline is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
  + Creating a virtual environment in terminal: `conda create -n Nerveseg python=3.7`.
    
  + Installing necessary packages: Tensorflow 1.13.1, Keras 2.3.0

  > Note that, the Official Source of Tensorflow 1.13.1 may not support the Ampere Structure GPU (NVIDIA RTX 30 Series), and you may need to download Tensorflow at the [source of NVIDIA](https://github.com/NVIDIA/tensorflow).

  + Or to apply the command: `pip install -r requirements.txt`


**2. Downloading necessary data:** 

  + downloading dataset (Training, Testing and Validating) and move it into `./data/`, 
    which can be found in this [download link (Onedrive)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EfUTJTiaiZdLs_-ZQYEoXwwBAW6GLyi0HGx4qyluNYeLXg?e=lkOcqL).
  
  + downloading external test set and move it into `./example_data/`, 
    which can be found in this [download link (Onedrive)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EXP9pL7rdR5Pg9PoVlAtujkBZWgD4cN7tPlC9WaTaLa3yA?e=f5cmos).
    
  + downloading classification weights and move it into root, 
    which can be found in this [download link (Onedrive)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/ESHTi0OQfEBDl0pTEd4EWyMBjXrKj0yxWaR4jUOHarH9Vw?e=4em0Mm).
    
  + downloading segmentation weights and move it into root,
    which can be found in this [download link (Onedrive))](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EXSUKzDdX_NLn8c0Q6phJmkBCBlgOvMlhvbkK6HhLLMmuQ?e=CHqZhs).

   
**3. Training Configuration:** 

  + Assigning your costumed path. For example:

    `--train_path` in `train.py` (Training Data Path),

    `--valid_path` in `train.py` (Validating Data Path),

    `model_path` in `train.py` (The saved Weight Path),
  
  + Configurate the setting. For example:

    `batch_size` in `train.py` (Batch Size),

    `epochs` in `train.py` (Round of Training),
    
    `lr` in `train.py` (Initial Learning Rate),    

    `shape` in `train.py` (The shape of input image),
    
  + Start Training!

**4. Testing Configuration:** 

  + Assigning your costumed path. For example:

    `--folder_name` in `link_copy.py` (External Test Set Path),

    `model_seg ` in `link_copy.py` (Saved Segmentation Weight),

    Result will be generated at `--folder_name + /output/`
    
  + Start Testing!


### 3.2 Evaluating your trained model:

Python: For Classification, we use commercial software to calculate metrics. For Segmentation, the evaluation code is `dice_iou.py`.

### 3.3 More Result: 

Coming Soon!

## 4. Citation

Please cite our paper if you find the work useful: 

## 5. TODO LIST

> If you want to improve the usability or any piece of advice, please feel free to contact me directly ([E-mail]()).

- [ ] Support more Classification Networks (
[ResNeXt](https://github.com/facebookresearch/ResNeXt), 
[ResNeSt](https://github.com/zhanghang1989/ResNeSt), 
[Vision Transformer](https://github.com/google-research/vision_transformer),
and 
[Swin Transformer](https://github.com/microsoft/Swin-Transformer) 
etc.)

- [ ] Support more Segmentation Networks.

- [ ] Support lightweight architecture and real-time inference, like MobileNet, SqueezeNet.

- [ ] Add more comprehensive competitors.

## 6. FAQ

Coming Soon!

## 7. License

The source code is free for research and education use only. Any comercial use should get formal permission first.

---
