# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot
from skimage import io,img_as_float
dir_path = '/home/silence/proj/photos'  #训练
photo_list = os.listdir(dir_path)
for photo in sorted(photo_list):
    try:
        im = img_as_float(io.imread(dir_path+'/'+photo))
    except ValueError as e:
        os.remove(dir_path+'/'+photo)
        print('remove truncated image:',photo)
    
