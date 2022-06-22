import os
import glob
import imageio
from natsort import natsorted, ns
import cv2
import numpy as np




png_dir = '/home/htihe/NerveSegmentation/Video_data/S6/output2/'

images = []
file_names = glob.glob(png_dir+"*.jpg")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.split("_")[-1].lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    images.append(file_name)
    print(file_name)

size=(1080,1080)

out = cv2.VideoWriter('/home/htihe/NerveSegmentation/Video_data/S60''/sample_v2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 18, size)

for i in range(len(images)):
    img = cv2.imread(images[i])
    #img = cv2.resize(img, (512, 512))
    #height, width, layers = img.shape
    #size = (width, height)
    out.write(img)
out.release()


# Opening wound, tendon, opennerve, nerve, trannerve
# Number