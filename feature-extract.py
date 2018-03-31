# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
# 设置当前的工作环境在caffe下
caffe_root = '/home/silence/caffe/caffe/'
# 把caffe/python也添加到当前环境
sys.path.insert(0, caffe_root + 'python')
import caffe
# 更换工作目录
os.chdir(caffe_root)
#caffe.set_mode_cpu() 
caffe.set_mode_gpu() #GPU模式运行要加sudo
# 设置网络结构
net_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
# 添加训练之后的参数
caffe_model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# 将上面两个变量作为参数构造一个net
net = caffe.Net(net_file, #定义模型结构
                caffe_model, #包含模型训练权值
                caffe.TEST) #使用测试模式(不执行dropout)
# 加载Imagenet的图像均值文件
mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
''' 
matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB  
caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换
'''
# channel放到前面 skimage读出来的是(height,width, channel) ，但caffe可以处理的图片的数据格式是(channel,height,width)
transformer.set_transpose('data',(2,0,1))
# 对于每个通道，都减去BGR的均值像素值
transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
# 图片像素放大到[0-255]
transformer.set_raw_scale('data',255)
# RGB -> BGR 转换
transformer.set_channel_swap('data',(2,1,0))
<<<<<<< HEAD
'''
=======
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
# 设置输入图像大小
net.blobs['data'].reshape(50,        # batch 大小
                          3,         # 3-channel (BGR) images
                          227, 227)  # 图像大小为:227x227
<<<<<<< HEAD
'''
=======
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
# 加载imagenet标签
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file,str,delimiter='\t')
# 加载图片
features = []
photo_names = []
<<<<<<< HEAD
#dir_path = '/home/silence/proj/photos'  #训练
dir_path = '/home/silence/proj/photos_test'  #测试
=======
dir_path = '/home/silence/proj/photos_demo'  #训练
#dir_path = '/home/silence/proj/photos_test_demo'  #测试
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
photo_list = os.listdir(dir_path)
for photo in sorted(photo_list):
    im = caffe.io.load_image(dir_path+'/'+photo)
    # 用之前的设定处理图片 
    transformed_image = transformer.preprocess('data',im)
    # 将图像数据拷贝到为net分配的内存中
    net.blobs['data'].data[...] = transformed_image
    # 网络向前传播
    output = net.forward()
    feat_fc7 = net.blobs['fc7'].data[0]
    features.append(feat_fc7.copy())
    photo_names.append(photo)
save_path = '/home/silence/proj/'
features = np.array(features)
### 训练
<<<<<<< HEAD
'''
=======

>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
# PCA
pca = PCA(n_components=512)
features_pca = pca.fit_transform(features) #用训练图片来训练PCA模型，同时返回降维后的数据
pca_model = open(save_path+'pca_model.pkl','wb') #保存训练好的pca模型
file_features = open(save_path+'features.pkl','wb') 
file_names = open(save_path+'photonames.pkl','wb')
pickle.dump(pca,pca_model)
pickle.dump(features_pca,file_features)
pickle.dump(photo_names,file_names)
<<<<<<< HEAD
'''
### 测试

=======

### 测试
'''
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
pca_model = open(save_path+'pca_model.pkl','rb')
pca = pickle.load(pca_model)
features_pca = pca.transform(features) #当模型训练好后，对于新输入的数据，用transform方法来降维
file_features = open(save_path+'test_features.pkl','wb')  #测试
file_names = open(save_path+'test_photonames.pkl','wb')
pickle.dump(features_pca,file_features)
pickle.dump(photo_names,file_names)
<<<<<<< HEAD

=======
'''
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658





