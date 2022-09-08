# Nerve_cls_seg

> **Authors:** 
> Alex Ngai Nick Wong, 
> Zebang He, 
> Jung Sun Yoo, 

- The Code for State-of-the-art Segmentation method shown in the figure is in Segmentation_SOTA branch.

## 1. Preface

- This repository provides code for "_**Nerve Classification and Segmentation**_" in State-of-the-art Segmentation method.

- If you have any questions about our paper, feel free to contact us. And if you are using the work for your research, please cite this paper ([BibTeX](#4-citation)).

- The methods included in segmentation_SOTA branch is listed below:

   + DeepLab V3+ (Backbone: ResNet, DRN, XCeption)
   
   + UNet

   + GateUNet

   + Transformer

- Please note that, the Double-UNet is `train_seg.py` in main branch, the MultiResUnet is `train_multiresunet.py` in main branch. 

## 2. Run the Code

 - Locate to the folder `Seg_rawcode`

 - Modify the subfolder in `utils_gray.py` (Grayscale image, line 130,131) or `utils.py` (RGB image, line 130,131)

 - Train Script is `train.sh`

   + Sample Script:

     - python train.py --train_dataset "Rearrange/train/" --val_dataset "Rearrange/val/" --direc 'test_unet2/' --batch_size 2 --epoch 102 --save_freq 20 --modelname "gatedaxialunet" --learning_rate 0.001 --imgsize 256 --gray "yes"

     - train_dataset is the path of training folder, val_dataset is the path of validation folder, direc is the folder to save the checkpoint.

     - batch_size, epoch, save_freq(The frequency to save model checkpoint), modelname(The training model type), learning-rate, imgsize(The size of the training images), gray(Grayscale or RGB images)
  
  - Test Script is `test.sh` and similar parameters with Train Script.


## 3. License

The source code is free for research and education use only. Any comercial use should get formal permission first.

---
