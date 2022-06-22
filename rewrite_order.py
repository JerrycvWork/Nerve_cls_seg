import glob
import shutil
import time





for filename in glob.glob(r"/home/htihe/NerveSegmentation/Video_data/S4/1113M2R/*"):
    shutil.copy(filename,r"/home/htihe/NerveSegmentation/Video_data/S4/annotation/Nerve/Mix_"+filename.split("/")[-2]+"__"+filename.split("/")[-1])