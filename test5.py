# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
from skimage import io,img_as_float,color
dir_path = '/home/silence/proj/photos/' 
image_list = os.listdir(dir_path)
image_list.sort()
img = io.imread(dir_path+image_list[64158])
plt.imshow(img)
plt.axis('off') # 不显示坐标轴
plt.show()
print(img)
lab = color.rgb2lab(img)
# Lab值归一化，L,a,b各分量原本范围:L[0,100], a,b[-128,127]
lab[:,:,0] = lab[:,:,0]/100
lab[:,:,1:3] = (lab[:,:,1:3]+128)/255

lab_vec = lab.reshape(lab.shape[0]*lab.shape[1],3) 
color_layer = lab_vec[:,0:3] ######
mu = np.mean(color_layer,axis=0) #axis = 0压缩行，对各列求均值，返回1*n矩阵
cov = np.cov(color_layer.transpose())
print(cov)
