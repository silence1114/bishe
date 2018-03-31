# -*- coding: utf-8 -*-
import os
import cv2
import pickle
import numpy as np
from scipy import optimize
from skimage import io, color,exposure,img_as_ubyte
from PIL import Image
import matplotlib.pyplot as plt
ref_path = '/home/silence/proj/style_ref/'
save_path = '/home/silence/proj/'
top_k = 3 #显示风格计分最高的k张参考图像
num_of_labels = 1000
if __name__ == '__main__':
    # 载入风格库图像文件名列表
    image_names_file = open(save_path+'ref-imagenames.pkl', 'rb')
    ref_imagenames = np.array(pickle.load(image_names_file))
    # 载入风格排序结果
    ranking_file = open(save_path+'style-ranking.pkl', 'rb')
    style_ranking_all = pickle.load(ranking_file)
    for label in range(num_of_labels):
        style_ranking = style_ranking_all[label].copy()
        top_styles = style_ranking.argsort()[::-1][:top_k] #[x:y:z]切片索引,x是左端,y是右端,z是步长,在[x,y)区间从左到右每隔z取值,默认z为1可以省略z参数.步长的负号就是反向,从右到左取值.
        ref_images = ref_imagenames[top_styles].copy()
        fig = plt.figure()
        i = 1
        for ref_image in ref_images:
            img = Image.open(ref_path+ref_image)
            ax = fig.add_subplot(1,top_k,i)
            ax.imshow(img)
            plt.axis('off')
            i += 1
        plt.show()
            
            
