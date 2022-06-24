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


### 2.1. Introduction (OLD)

In vivo SHG reflectance imaging of the sciatic nerve, the nerve-specific reflectance video is often recorded for further analysis in surgery. In clinical practice, segmenting and identifying the nerve from the nerve-specific reflectance video is of great importance since it provides valuable information for diagnosis and surgery. However, accurately segmenting out the nerve is a challenging task, for three major reasons: (i) the dataset with fine annotations is lack for deep learning-based application; (ii) the frames may not contains the nerves and only the segmentation network may hard to avoid false segmenting; (iii) nerves have a diversity of size, shape, and location.

To address these challenges, it is proposed the nerve-specific reflectance video dataset with labels of surgery stages and nerve masks and a pipeline for accurate surgery stage identification and nerve segmentation in reflectance video. Specifically, the frames of video are input to the pipeline and get the labels of the surgery stage (General Field, Tendon, and Nerve) through DenseNet201. For the frames that contain nerves, the pipeline then generates the nerve masks by the DoubleUNet segmentation network. 

Quantitative and qualitative evaluations on training, testing, and validating sets across metrics in classification and segmentation show that our pipeline improves the classification and segmentation accuracy significantly for the nerve-specific reflectance video.


### 2.2. Introduction (New)

In this study, we describe the development of intraoperative imaging system through purposeful screening the possible wavelength for peripheral nerve identification and the design of real-scenario inspired pipeline for nerve detection, resulting to spotlight nerve immediately during surgery. We used BALB/c mice to perform spectroscopic analysis to measure the reflectance using two-photon microscopy. The second-harmonic generation (SHG) signal generated from two-photon microscopy can allow deep tissue imaging with reduction of light scattering signal as reflectance light signal. Nerve and its surrounding tissues including muscle, fat, tendon, and vein were screened for their reflectance properties under ex vivo and in vivo condition using two-photon microscopy to calculate nerve specificity.  Nerve specificity evaluation involved to nerve to background tissue ratio (NBR) quantification and usually compared with muscle and adipose tissue, while greater value of NBR demonstrate effective nerve specificity. Imaging performance comparison between conventional fluorescent myelin stains and nerve-specific reflectance were conducted based on nerve specificity using two-photon microscopy. We had further showcased intraoperative imaging performance to identify nerve using two different murine cancer models by subcutaneous injection of malignant tumour cells from 4T1-luc2-RFP breast cancer and K562-GFP cell line. Spectral imaging performance had also showcased by 4T1 murine cancer model using stereomicroscope equipped with customized nerve-specific reflectance optical filter to mimic current optical setting used in FGS. Finally, we displayed our automated pipeline using nerve specific reflectance video imaging to achieve simultaneous nerve detection and demonstrate the potential clinical utility. We firstly fine-tuned DenseNet201 model to recognize all video frames containing nerve, then nerve delineation would be performed from the previously recognized video frames containing nerve by the optimized DoubleUNet to highlight with pseudo-color. Taking advantage of growing varieties of deep learning algorithms, comparative analysis for different models to recognize nerve anatomical structure and quantify the exact nerve location will also conducted.  


### 2.3. Framework Overview

<p align="center">
    <img src="Readme_figure\pipeline.png"/> <br />
    <em> 
    Figure 1: Schematic diagram of the multi-task deep learning-based nerve imaging system to process the nerve-specific reflectance video recording.
    </em>
</p>


### 2.4. Qualitative Results

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

**3.1. Classification Model:** 
  + Assigning your costumed path. For example:

    `train_dir` in `train_cls.py` (Training Data Path),

    `validation_dir` in `train_cls.py` (Validating Data Path),

    `model_path` in `train_cls.py` (The saved Weight Path),
  
  + Configurate the setting. For example:

    `BATCH_SIZE` in `train_cls.py` (Batch Size),

    `initial_epochs`,`fine_tune_epochs` in `train_cls.py` (Round of Training),
    
    `base_learning_rate` in `train_cls.py` (Initial Learning Rate),    

    `IMG_SIZE` in `train_cls.py` (The shape of input image),
    
  + Start Training!

**3.2. Segmentation Model:** 
  + Assigning your costumed path. For example:

    `--train_path` in `train_seg.py` (Training Data Path),

    `--valid_path` in `train_seg.py` (Validating Data Path),

    `model_path` in `train_seg.py` (The saved Weight Path),
  
  + Configurate the setting. For example:

    `batch_size` in `train_seg.py` (Batch Size),

    `epochs` in `train_seg.py` (Round of Training),
    
    `lr` in `train_seg.py` (Initial Learning Rate),    

    `shape` in `train_seg.py` (The shape of input image),
    
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
