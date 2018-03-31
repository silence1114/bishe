# -*- coding: utf-8 -*-
import os
import cv2
import pickle
<<<<<<< HEAD
from PIL import Image
import pandas as pd
=======
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
import numpy as np
from scipy import optimize
from skimage import io, color,exposure,img_as_ubyte
import matplotlib.pyplot as plt
<<<<<<< HEAD
dir_path = '/home/silence/proj/photos_test/'
save_path = '/home/silence/proj/'
ref_path = '/home/silence/proj/style_ref/'
top_k = 3 #取风格计分最高的k张参考图像进行转换
gamma = 2.2 
def style_trans(input_im,ref_im):
    faces = face_classify(input_im)
=======
dir_path = '/home/silence/proj/photos_test_demo/'
save_path = '/home/silence/proj/'
ref_path = '/home/silence/proj/style_ref_demo/'
top_k = 3 #取风格计分最高的k张参考图像进行转换
gamma = 2.2 
def style_trans(input_im,ref_im):
  
    plt.imshow(input_im)
    plt.axis('off') # 不显示坐标轴
    plt.show()
   

>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    input_im_lab = pre_process(input_im)
    ref_im_lab = pre_process(ref_im)
    output_im = input_im_lab.copy()
    output_im_color = color_transfer(input_im_lab,ref_im_lab,output_im)
<<<<<<< HEAD
    output_im_luminance = luminance_transfer(input_im_lab.copy(),ref_im_lab,output_im_color.copy())
    output_im_corr = face_correct(output_im_luminance,faces)
    output_image = post_process(output_im_corr)

    return output_image.copy()
    

# 预处理
def pre_process(origin_image):
    image_lab = color.rgb2lab(origin_image)
=======
    output_im_luminance = luminance_transfer(output_im_color.copy(),ref_im_lab,output_im_color.copy())
    face_correct(output_im_luminance)
    '''
    output_image = post_process(output_im_luminance)
    
    plt.imshow(output_image)
    plt.axis('off') # 不显示坐标轴
    plt.show()
    '''
   

# 预处理
def pre_process(origin_image):
    # Gamma校正
    image_gamma = exposure.adjust_gamma(origin_image,gamma)
    # 转换到CIELab颜色空间
    image_lab = color.rgb2lab(image_gamma)
    # 取出亮度通道
    image_luminance = image_lab[:,:,0]
    image_luminance_flat = image_luminance.flatten()
    luminance_min = min(image_luminance_flat)
    luminance_max = max(image_luminance_flat)
    # 亮度级拉伸
    image_luminance = (image_luminance - luminance_min) / (luminance_max - luminance_min) * 100
    image_lab[:,:,0] = image_luminance.copy()
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    return image_lab


# 颜色变换
def color_transfer(input_im,ref_im,output_im):
    # 矩阵转化为三维向量，也就是一个h×w行3列的矩阵
    input_vec = input_im.reshape(input_im.shape[0]*input_im.shape[1],3) #shape[0] 图片宽度 shape[1] 图片高度 shape[2] 图片通道数
    ref_vec = ref_im.reshape(ref_im.shape[0]*ref_im.shape[1],3)
    output_vec = output_im.reshape(output_im.shape[0]*output_im.shape[1],3)
    # 提取色彩层
<<<<<<< HEAD
    input_color_layer = input_vec[:,1:3] 
    ref_color_layer = ref_vec[:,1:3]
=======
    input_color_layer = input_vec[:,0:3] 
    ref_color_layer = ref_vec[:,0:3]
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    # 求均值
    input_mu = np.mean(input_color_layer,axis=0) #axis = 0压缩行，对各列求均值，返回1*n矩阵
    ref_mu = np.mean(ref_color_layer,axis=0)
    # 求方差
    input_cov = np.cov(input_color_layer.transpose()) #先转置变成2行h×w列的矩阵，然后cov将一列视为一个变量，因此有h×w个2维变量，输出一个2×2的协方差矩阵，其中对角线元素是每个维度的方差，非对角线上的元素则是不同维度间的协方差。
    ref_cov = np.cov(ref_color_layer.transpose())
<<<<<<< HEAD
    
    # 正则化
    #lambda_r = 7.5 #正则化系数
    #input_cov = np.maximum(input_cov,np.eye(input_cov.shape[0])*lambda_r)
=======
    # 正则化
    lambda_r = 7.5 #正则化系数
    input_cov = np.maximum(input_cov,np.eye(input_cov.shape[0])*lambda_r)
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    # 求颜色变换矩阵 
    # 其中求矩阵的-1/2次方的方法参考：http://blog.csdn.net/jiangjieqazwsx/article/details/45675859 注意该方法中矩阵的特征分解式子写错了，应为Q*lambda*Q^-1
    # 求特征值 特征向量
    eigVals_input,eigVects_input = np.linalg.eig(input_cov)# eigVals存放特征值，行向量。eigVects存放特征向量，每一列带别一个特征向量。特征值和特征向量是一一对应的
    eigVects_input = np.matrix(eigVects_input)
    # 构造特征值对角矩阵的1/2次方
    diag_input = np.matrix(np.diag(np.sqrt(eigVals_input))) #diag()返回一个矩阵的对角线元素，或者创建一个对角阵
    # 构造协方差矩阵的1/2次方
    input_sigma1 = np.matrix(np.dot(np.dot(eigVects_input,diag_input),eigVects_input.I))
    # 构造协方差矩阵的-1/2次方
    input_sigma2 = input_sigma1.I
    # 构造中间结果的1/2次方
    tmp = np.matrix(np.dot(np.dot(input_sigma1,ref_cov),input_sigma1))
    eigVals_tmp,eigVects_tmp = np.linalg.eig(tmp)
    eigVects_tmp = np.matrix(eigVects_tmp)
    diag_tmp = np.diag(np.sqrt(eigVals_tmp))
    tmp2 = np.matrix(np.dot(np.dot(eigVects_tmp,diag_tmp),eigVects_tmp.I))
    # 求变换矩阵
    trans_mat = np.matrix(np.dot(np.dot(input_sigma2,tmp2),input_sigma2))
    # 验证计算结果 两者相等就正确
    #tmp3 = np.dot(np.dot(trans_mat,input_cov),trans_mat.transpose())
    #print(tmp3)
    #print(ref_cov)
<<<<<<< HEAD
   
    # 进行色彩变换
    output_color_layer = np.dot(trans_mat,(input_color_layer-input_mu).T).T+ref_mu
    output_vec[:,1:3] = output_color_layer.copy() 
    output_image = output_vec.reshape(output_im.shape[0],output_im.shape[1],3)
    
=======
    '''
    # 原方法
    sigma_image = input_cov.copy()
    sigma_sample = ref_cov.copy()
    [val_i, vec_i] = np.linalg.eig(sigma_image)
    val_i[val_i < 0] = 0
    da = np.diag(np.sqrt(val_i + np.finfo(float).eps))
    c = np.matrix(da) * (np.matrix(vec_i).T) * np.matrix(sigma_sample) * np.matrix(vec_i) * np.matrix(da)
    [val_c, vec_c] = np.linalg.eig(c)
    val_c[val_c < 0] = 0
    dc = np.diag(np.sqrt(val_c + np.finfo(float).eps))
    da_inv = np.diag(1 / (np.diag(da)))
    mm = (np.matrix(vec_i) * np.matrix(da_inv)) * \
               (np.matrix(vec_c) * np.matrix(dc) * (np.matrix(vec_c).T)) * \
               (np.matrix(da_inv) * (np.matrix(vec_i).T))
    print(mm)
    print(trans_mat)
    '''
    # 进行色彩变换
    
    #output_color_layer = np.dot((input_color_layer-input_mu),trans_mat)+ref_mu
    output_color_layer = np.dot(trans_mat,(input_color_layer-input_mu).T).T+ref_mu
    output_vec[:,0:3] = output_color_layer.copy()
    #output_vec[:,0] = input_vec[:,0]
    output_image = output_vec.reshape(output_im.shape[0],output_im.shape[1],3)
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    return output_image.copy()

# 亮度变换
def luminance_transfer(input_im,ref_im,output_im):
<<<<<<< HEAD
    epsilon = 0.5
    num_of_samples = 32 # 采样数/特征维数
    # 提取亮度特征
    # 求亮度层直方图 
    input_luminance = input_im[:,:,0].flatten()
    #input_luminance[input_luminance==np.NaN] = 0
    #input_luminance = pd.DataFrame(input_im[:,:,0].flatten()).fillna(0)
    #input_luminance = np.array(input_luminance)
    hist_input,bin_edges_input = np.histogram(input_luminance,bins=256,normed=True) #当normed参数为False时，函数返回数组a中的数据在每个区间的个数，等于true时对个数进行正规化处理，使它等于每个区间的概率密度
=======
    tau = 0.4
    num_of_samples = 32 # 采样数/特征维数
    # 提取亮度特征
    # 求亮度层直方图 
    hist_input,bin_edges_input = np.histogram(input_im[:,:,0].flatten(),bins=256,normed=True) #当normed参数为False时，函数返回数组a中的数据在每个区间的个数，等于true时对个数进行正规化处理，使它等于每个区间的概率密度
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    hist_ref,bin_edges_ref = np.histogram(ref_im[:,:,0].flatten(),bins=256,normed=True) 
    # 累积分布函数 cumulative distribution function
    cdf_input = hist_input.cumsum()
    cdf_ref = hist_ref.cumsum()
    # 归一化
    dx_input = bin_edges_input[1] - bin_edges_input[0] #bin_edges数组长度为len(hist)+1,每两个相邻的数值构成一个统计区间
    dx_ref = bin_edges_ref[1] - bin_edges_ref[0]
    cdf_input = cdf_input * dx_input
    cdf_ref = cdf_ref * dx_ref
    #plt.plot(bin_edges_ref[1:], hist_ref)
    #plt.show()
    # 对Y轴等分
    dy_input = np.arange(1,1+num_of_samples)/num_of_samples
    dy_ref = np.arange(1,1+num_of_samples)/num_of_samples
    # 二分法查找Y轴等分点在X轴上的位置，返回位置索引值
    index_input = np.searchsorted(cdf_input,dy_input)
    index_ref = np.searchsorted(cdf_ref,dy_ref)
    # 获取对应的X轴上的坐标作为亮度特征
    input_luminance_feature = bin_edges_input[index_input]
    ref_luminance_feature = bin_edges_ref[index_ref]
<<<<<<< HEAD
    #tmp_luminance_feature = input_luminance_feature + (ref_luminance_feature-input_luminance_feature)*(tau/max(tau,np.linalg.norm(ref_luminance_feature-input_luminance_feature,ord=np.inf)))
    tmp_luminance_feature = input_luminance_feature + (ref_luminance_feature-input_luminance_feature)*epsilon
    print(np.linalg.norm(ref_luminance_feature-input_luminance_feature,ord=np.inf))
    # 最小化代价函数求参数
    cost_func = lambda param : np.power(np.linalg.norm((np.arctan(param[0]/param[1])+np.arctan((input_luminance_feature-param[0])/param[1]))/(np.arctan(param[0]/param[1])+np.arctan((1-param[0])/param[1]))-tmp_luminance_feature ,2),2)
    #(np.arctan(param[0]/param[1])+np.arctan((input_luminance_feature-param[0])/param[1]))/(np.arctan(param[0]/param[1])+np.arctan((1-param[0])/param[1])) # 转换函数
    res = optimize.minimize(cost_func,x0=np.random.random_sample(2),method = 'Nelder-Mead')# 不同method可能导致不收敛
    #res = optimize.minimize(cost_func,x0=np.random.random_sample(2),method = 'Nelder-Mead',options={'disp': True}) 
=======
    tmp_luminance = input_luminance_feature + (ref_luminance_feature-input_luminance_feature)*(tau/min(tau,np.linalg.norm(ref_luminance_feature-input_luminance_feature,ord=np.inf)))
    # 最小化代价函数求参数
    cost_func = lambda param : np.power(np.linalg.norm((np.arctan(param[0]/param[1])+np.arctan((input_luminance_feature-param[0])/param[1]))/(np.arctan(param[0]/param[1])+np.arctan((1-param[0])/param[1]))-tmp_luminance,2),2)
    #(np.arctan(param[0]/param[1])+np.arctan((input_luminance_feature-param[0])/param[1]))/(np.arctan(param[0]/param[1])+np.arctan((1-param[0])/param[1])) # 转换函数
    res = optimize.minimize(cost_func,x0=np.random.random_sample(2),method = 'Nelder-Mead',options={'disp': True}) # 不同方法可能导致不收敛
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    param = res.x
    # 亮度变换
    output_im[:,:,0] = (np.arctan(param[0]/param[1])+np.arctan((input_im[:,:,0]-param[0])/param[1]))/(np.arctan(param[0]/param[1])+np.arctan((1-param[0])/param[1]))
    return output_im

<<<<<<< HEAD
# 面部识别
def face_classify(input_im):
    input_im_copy = input_im.copy()
   
    # 提取亮度通道
    input_luminance = input_im[:,:,0].copy()
    # 面部识别
    face_classifier = cv2.CascadeClassifier('/home/silence/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml') #定义分类器
    faces = face_classifier.detectMultiScale(img_as_ubyte(input_luminance / np.max(input_luminance)), 1.3, 5)#人脸检测  # 参数介绍：image--待检测图片，一般为灰度图像加快检测速度；scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%; minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框
    if len(faces)>0:#如果人脸数组长度大于0
        for face in faces: 
                x, y, w, h = face
                #cv2.rectangle(input_im_copy, (x, y), (x+w, y+h), (0,0,0))#对每一个人脸画矩形框
                
    #plt.imshow(input_im_copy)
    #plt.axis('off') # 不显示坐标轴
    #plt.show()
    return faces

# 面部校正
def face_correct(input_im,faces):
    # 参数
    luminance_th = 0.6
    epsilon = 1.5
    gamma_th = 0.5
    alpha_r = 3
    alpha_c = 0.0005
    output_im = input_im.copy()
    for face in faces:
        x,y,w,h = face
        # 获取面部矩形区域
        btm = y+h
        top = y
        left = x
        right = x+w
        center = [y+h/2,x+w/2]
        radius = max(w,h)/2
        im_box = input_im[top:btm,left:right,:]
        im_box_luminance = im_box[:,:,0]
        im_box_luminance = im_box_luminance/100
        mask = im_box_luminance.copy()
        mask[mask>=0] = 1
        mask[mask<0] = 0
        # 获取亮度信息
        luminance_max = np.max(im_box_luminance)
        luminance_min = np.min(im_box_luminance)
        luminance_med = np.median(im_box_luminance)
        box_vec = im_box.reshape(im_box.shape[0]*im_box.shape[1],3)
        color_med = np.median(box_vec, axis=0)
        #luminance_th = (luminance_max - luminance_min) * luminance_th_r + luminance_min
        print('luminance_med: ',luminance_med)
        
        # 亮度补偿
        if luminance_med < luminance_th:
           print('doing luminance correction')
           gamma_corr = max(gamma_th, luminance_med / luminance_th)
           [row, col] = np.mgrid[top:btm,left:right]
           w1 = np.exp(-alpha_r*(np.power(row-center[0],2)+np.power(col- center[1],2))/np.power(radius, 2))
           [row, col] = np.mgrid[top:btm,left:right]
           tmp = np.power(im_box[row-top,col-left,0]-color_med[0],2)+np.power(im_box[row-top,col-left,1]-color_med[1],2)+np.power(im_box[row-top,col-left,2]-color_med[2],2)
           w2 = np.exp(-alpha_c*tmp)
           w = epsilon*w1*w2
           tmp_box_luminance = im_box_luminance.copy()
           tmp_box_luminance[tmp_box_luminance<0] = 0
           luminance_corr = ((1-w)*tmp_box_luminance+w
*(np.power(tmp_box_luminance, gamma_corr)))*mask+(1-mask)*im_box_luminance
           luminance_corr = luminance_corr*100
           output_im[top:btm,left:right,0] = luminance_corr.copy()
          
    return output_im.copy()
=======
# 面部校正
def face_correct(input_im):
    # 参数
    luminance_th = 0.5
    gamma_th = 0.5
    alpha_r = 0.45
    alpha_c = 0.001
    # 提取亮度通道
    input_luminance = input_im[:,:,0]
    # 面部识别
    face_classifier = cv2.CascadeClassifier('/home/silence/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml') #定义分类器
    faces = face_classifier.detectMultiScale(img_as_ubyte(input_luminance / np.max(input_luminance)), 1.2, 2)#人脸检测
    if len(faces)>0:#如果人脸数组长度大于0
        for face in faces: #对每一个人脸画矩形框
                x, y, w, h = face
                print(x, y, w, h)
    
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658


# 后处理
def post_process(output_im):
    image_rgb = color.lab2rgb(output_im)
<<<<<<< HEAD
    return image_rgb
    
   
=======
    image_gamma = exposure.adjust_gamma(image_rgb,1/gamma)
    return image_gamma
    
   

>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
if __name__ == '__main__':
    # 载入测试图片的分类结果
    test_labels = open(save_path+'test_labels.pkl', 'rb')
    labels = pickle.load(test_labels)
    # 载入风格库图像文件名列表
    image_names_file = open(save_path+'ref-imagenames.pkl', 'rb')
    ref_imagenames = np.array(pickle.load(image_names_file))
    # 载入风格排序结果
    ranking_file = open(save_path+'style-ranking.pkl', 'rb')
    style_ranking_all = pickle.load(ranking_file)
    photo_list = os.listdir(dir_path)
    index = 0
    for photo in sorted(photo_list):
<<<<<<< HEAD
        input_im = io.imread(dir_path+photo)
        fig = plt.figure()
        im = Image.open(dir_path+photo)
        ax = fig.add_subplot(2,4,1)
        ax.imshow(im)
        ax.set_title('Input image')
        plt.axis('off')
=======
        #input_im = io.imread(dir_path+photo)
        input_im = io.imread('/home/silence/proj/photos_test_demo/20160512005366.jpg')
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
        label = labels[index]
        style_ranking = style_ranking_all[label].copy()
        top_styles = style_ranking.argsort()[::-1][:top_k] #[x:y:z]切片索引,x是左端,y是右端,z是步长,在[x,y)区间从左到右每隔z取值,默认z为1可以省略z参数.步长的负号就是反向,从右到左取值.
        ref_images = ref_imagenames[top_styles].copy()
<<<<<<< HEAD
        i = 2
        for ref_image in ref_images:
            ref_im = io.imread(ref_path+ref_image)
            output_im = style_trans(input_im,ref_im)
            img = Image.open(ref_path+ref_image)
            ax = fig.add_subplot(2,4,i)
            ax.imshow(img)
            if i==3:
                ax.set_title('Reference image')
            plt.axis('off')
            ax = fig.add_subplot(2,4,i+4)
            ax.imshow(output_im)
            if i==3:
                ax.set_title('Result')
            plt.axis('off')
            i += 1
            
        plt.show()
        index += 1
       

=======
        for ref_image in ref_images:
            #ref_im = io.imread(ref_path+ref_image)
            ref_im = io.imread('/home/silence/proj/style_ref_demo/good-001.jpg')
            style_trans(input_im,ref_im)
            #output_im = style_trans(input_im,ref_im)
            #io.imshow(output_im)
            break #
        break #
        index += 1
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
