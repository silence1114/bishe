# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
from skimage import io, color
import matplotlib.pyplot as plt
from multiprocessing import Pool,Manager,Process
dir_path = '/home/silence/proj/style_ref/' #风格图像
#dir_path = '/home/silence/proj/photos/'   #内容图像
save_path = '/home/silence/proj/'
def extract_style_feature(image_index,filename,luminance_features,color_mu,color_cov):
    # 加载rgb图像
    rgb = io.imread(dir_path+filename)
    # 转换到CIELab色域
    lab = color.rgb2lab(rgb)
    # Lab值归一化，L,a,b各分量原本范围:L[0,100], a,b[-128,127]
    lab[:,:,0] = lab[:,:,0]/100
    lab[:,:,1:3] = (lab[:,:,1:3]+128)/255
    #print(np.array(lab[:,:,0]).flatten().tolist())
    ######################求亮度特征#########################
    # 求亮度层直方图 
    hist,bin_edges = np.histogram(lab[:,:,0].flatten(),bins=256,normed=True) #当normed参数为False时，函数返回数组a中的数据在每个区间的个数，等于true时对个数进行正规化处理，使它等于每个区间的概率密度
    # 累积分布函数 cumulative distribution function
    cdf = hist.cumsum()
    # 归一化
    dx = bin_edges[1] - bin_edges[0] #bin_edges数组长度为len(hist)+1,每两个相邻的数值构成一个统计区间
    cdf = cdf * dx
    #print(cdf)
    #plt.plot(bin_edges[1:], cdf)
    #plt.show()
    # 对Y轴等分
    num_of_samples = 32 #采样数
    dy = np.arange(1,1+num_of_samples)/num_of_samples
    # 二分法查找Y轴等分点在X轴上的位置，返回位置索引值
    index = np.searchsorted(cdf,dy)
    # 获取对应的X轴上的坐标作为亮度特征
    luminance_feature = bin_edges[index]
    #print(luminance_feature)
    ##########################################################

    #######################求颜色特征##########################
    # 矩阵转化为三维向量，也就是一个h×w行3列的矩阵
    lab_vec = lab.reshape(lab.shape[0]*lab.shape[1],3) #shape[0] 图片宽度 shape[1] 图片高度 shape[2] 图片通道数
    # 提取色彩层
    color_layer = lab_vec[:,0:3] ######
    # 求均值
    mu = np.mean(color_layer,axis=0) #axis = 0压缩行，对各列求均值，返回1*n矩阵
    # 求方差
    cov = np.cov(color_layer.transpose()) #先转置变成2行h×w列的矩阵，然后cov将一列视为一个变量，因此有h×w个2维变量，输出一个2×2的协方差矩阵，其中对角线元素是每个维度的方差，非对角线上的元素则是不同维度间的协方差。
    ###########################################################
    luminance_features[image_index] = luminance_feature.copy()
    color_mu[image_index] = mu.copy()
    color_cov[image_index] = cov.copy()
    print('Image %d done' % image_index)

if __name__ == '__main__':
    # 存储图片名列表
    image_list = os.listdir(dir_path)
    image_list.sort()
    image_names_file = open(save_path+'ref-imagenames.pkl', 'wb') #风格图像
    #image_names_file = open(save_path+'data-imagenames.pkl', 'wb') #内容图像
    pickle.dump(image_list,image_names_file)
    num_of_images = len(image_list)
    # 主进程与子进程共享这些list  
    luminance_features = Manager().list(range(num_of_images))
    color_mu = Manager().list(range(num_of_images))
    color_cov = Manager().list(range(num_of_images))
    # 多进程
    p = Pool(4) #开辟进程池
    for i in range(num_of_images):
        p.apply_async(extract_style_feature,args=(i,image_list[i],luminance_features,color_mu,color_cov))#每个进程都调用extract_style_feature函数，args表示给该函数传递的参数
       
    p.close() #关闭进程池
    p.join() #等待开辟的所有进程执行完后，主进程才继续往下执行
    # 储存风格特征
    style_features = open(save_path+'ref-style-features.pkl', 'wb') #风格图像
    #style_features = open(save_path+'data-style-features.pkl', 'wb') #内容图像
    pickle.dump({'luminance_features': list(luminance_features),
                 'color_mu': list(color_mu),
                 'color_cov':list(color_cov),}, style_features)## 注意，共享list不可直接dump，会报类型错误，必须先转换为普通list
    
        
    
        
