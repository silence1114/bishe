from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from math import isnan,ceil
import pickle
import numpy as np
from PIL import Image
from skimage import io, color,exposure,img_as_ubyte
import matplotlib.pyplot as plt
dir_path = '/home/silence/test/'
ref_path = '/home/silence/test/style_ref/'

def compute_distance(mu_ref1,cov_ref1,mu_ref2,cov_ref2,i,j):
    tmp1 = np.power(np.linalg.norm(mu_ref1 - mu_ref2, 2), 2)
    #tmp2 = np.trace(cov_ref1+cov_ref2-2*np.sqrt(np.abs(cov_ref1.dot(cov_ref2))))
    tmp2 = np.trace(cov_ref1.dot(cov_ref1)+cov_ref2.dot(cov_ref2)-2*(cov_ref1.dot(cov_ref2)))
    #ans = np.power(tmp1+tmp2,2)
    ans = np.sqrt(tmp1+tmp2)
    return np.exp(-ans)
'''
def compute_distance3(luminance_ref1,mu_ref1,cov_ref1,luminance_ref2,mu_ref2,cov_ref2,i,j):
    # 调节参数
    lambda_l = 0.005
    lambda_c = 0.05
    # 求亮度特征之间的欧氏距离
    de = np.linalg.norm(luminance_ref1 - luminance_ref2) 
    # 异常处理（有些图片只有单通道引起程序异常，忽略它们
    try: 
        tmp = np.linalg.det(cov_ref1)
        tmp = np.linalg.det(cov_ref2)
    except np.linalg.linalg.LinAlgError as e:
        if i==j:
            return 1
        else:
            return 0 
    if np.linalg.det((cov_ref1+cov_ref2)/2)<0 or np.linalg.det(cov_ref1)<0 or np.linalg.det(cov_ref2)<0:
        if i==j:
            return 1
        else:
            return 0    
    # 求色彩特征之间的Hellinger距离 (已经是平方形式
    tmp1 = np.power(np.linalg.det(cov_ref1),1/4)*np.power(np.linalg.det(cov_ref2),1/4)/np.power(np.linalg.det((cov_ref1+cov_ref2)/2),1/2)
    tmp2 = (-1/8)*np.dot(np.dot((mu_ref1-mu_ref2),((cov_ref1+cov_ref2)/2)),np.transpose(mu_ref1-mu_ref2))
    dh = 1 - (tmp1*np.exp(tmp2))
    # 计算相似度
    similarity = np.exp(-1*(np.power(de,2)/lambda_l))*np.exp(-1*(dh/lambda_c))
    return similarity

def compute_distance(mu_ref1,cov_ref1,mu_ref2,cov_ref2,i,j):
    lambda_c = 0.05
    # 异常处理（有些图片只有单通道引起程序异常，忽略它们
    try: 
        tmp = np.linalg.det(cov_ref1)
        tmp = np.linalg.det(cov_ref2)
    except np.linalg.linalg.LinAlgError as e:
        if i==j:
            return 1
        else:
            return 0  
    if np.linalg.det((cov_ref1+cov_ref2)/2)<0 or np.linalg.det(cov_ref1)<0 or np.linalg.det(cov_ref2)<0:
        if i==j:
            return 1
        else:
            return 0
    # 求色彩特征之间的Hellinger距离 (平方形式
    tmp1 = np.power(np.linalg.det(cov_ref1),1/4)*np.power(np.linalg.det(cov_ref2),1/4)/np.power(np.linalg.det((cov_ref1+cov_ref2)/2),1/2) 
    tmp2 = (-1/8)*np.dot(np.dot((mu_ref1-mu_ref2),((cov_ref1+cov_ref2)/2)),np.transpose(mu_ref1-mu_ref2))   
    dh = 1.0 - (tmp1*np.exp(tmp2))
    
    return np.exp(-1*(dh/lambda_c))
'''

if __name__ == '__main__':
    # 加载文件名列表
    ref_names_file = open(dir_path+'ref-imagenames.pkl', 'rb')
    ref_names = pickle.load(ref_names_file)
    num_of_ref = len(ref_names)
    # 加载特征文件
    ref_style_features_file = open(dir_path+'ref-style-features.pkl', 'rb')
    ref_style_features = pickle.load(ref_style_features_file)
    ref_luminance = ref_style_features['luminance_features']
    ref_mu = ref_style_features['color_mu']
    ref_cov = ref_style_features['color_cov']
    # 提前计算好距离
    distance = np.zeros((num_of_ref,num_of_ref))
    for i in range(num_of_ref):
        for j in range(num_of_ref):
            distance[i][j] = compute_distance(ref_mu[i],ref_cov[i],ref_mu[j],ref_cov[j],i,j)
           

    #distance[abs(distance)<1e-15] = 0
   
    pref = np.median(distance)
    af = AffinityPropagation(affinity='precomputed', preference=pref).fit(distance)
    cluster_centers_indices = af.cluster_centers_indices_ # 得到聚类的中心
    labels = af.labels_ # 得到label
    n_clusters = len(cluster_centers_indices) # 类的数目
    print(cluster_centers_indices, labels,n_clusters)
    names = np.array(ref_names)
    index_list = []
    names_list = []
 
    for label in range(n_clusters):
        index = np.where(labels == label)[0]
        index.sort()
        index_list.append(index.copy())
        names_list.append(names[index].copy())
    '''
    for label in range(n_clusters):
        print(label)
        fig = plt.figure()
        l = len(index_list[label])
        for i in range(l):
            img = Image.open(ref_path+names_list[label][i])
            ax = fig.add_subplot(3,ceil(l/3),i+1)
            ax.imshow(img)
            plt.axis('off')
        plt.show()
    '''
    file_index = open(dir_path+'ref-index.pkl','wb')
    pickle.dump(index_list,file_index)
    clustered_filenames = open(dir_path+'ref-clusteredNames.pkl','wb')
    pickle.dump(names_list,clustered_filenames)

