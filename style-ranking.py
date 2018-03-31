# -*- coding: utf-8 -*-
import numpy as np
import pickle
save_path = '/home/silence/proj/'
def compute_similarity(luminance_data,mu_data,cov_data,luminance_ref,mu_ref,cov_ref):
    # 调节参数
    lambda_l = 0.005
    lambda_c = 0.05
    # 求亮度特征之间的欧氏距离
<<<<<<< HEAD
    de = np.linalg.norm(luminance_data - luminance_ref) 
    # 异常处理（有些图片只有单通道引起程序异常，忽略它们
    try: 
        tmp = np.linalg.det(cov_data)
    except np.linalg.linalg.LinAlgError as e:
        return 0
    if np.linalg.det((cov_data+cov_ref)/2)<0 or np.linalg.det(cov_data)<0:
        return 0   
=======
    de = np.linalg.norm(luminance_data - luminance_ref)    
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    # 求色彩特征之间的Hellinger距离 (已经是平方形式
    tmp1 = np.power(np.linalg.det(cov_data),1/4)*np.power(np.linalg.det(cov_data),1/4)/np.power(np.linalg.det((cov_data+cov_ref)/2),1/2)
    tmp2 = (-1/8)*np.dot(np.dot((mu_data-mu_ref),((cov_data+cov_ref)/2)),np.transpose(mu_data-mu_ref))
    dh = 1 - (tmp1*np.exp(tmp2))
    # 计算相似度
    similarity = np.exp(-1*(np.power(de,2)/lambda_l))*np.exp(-1*(dh/lambda_c))
    
    return similarity
    '''
    epsilon = 1
    mu = np.matrix(np.abs(mu_data - mu_ref) + epsilon).T
    sigma = np.matrix((cov_data + cov_ref) / 2)
    dh_1 = np.power(np.linalg.norm(cov_ref.dot(cov_data), 1), 1 / 4) / (np.power(np.linalg.norm(sigma, 1), 1 / 2))
    dh_2 = (-1 / 8) * mu.T * np.linalg.inv(sigma) * mu
    dh = 1 - dh_1 * np.exp(dh_2)
    ans = np.exp(-np.power(de, 2) / lambda_l) * np.exp(-np.power(dh, 2) / lambda_c)
    return np.max(ans)
    '''


def scoring(i,j,k):
    similarity = compute_similarity(data_luminance[k],data_mu[k],data_cov[k],ref_luminance[j],ref_mu[j],ref_cov[j])
    
    if similarity>1e-10:
        score[i][j] += 10
    elif similarity<=1e-10 and similarity>1e-20:
        score[i][j] += 7
    elif similarity<=1e-20 and similarity>1e-40:
        score[i][j] += 4
    elif similarity<=1e-40 and similarity>1e-70:
        score[i][j] += 2
    elif similarity<=1e-70 and similarity>1e-100:
        score[i][j] += 1
    else:
        score[i][j] +=0
    
    

if __name__ == '__main__':
    # 加载文件名列表
    data_names_file = open(save_path+'data-imagenames.pkl', 'rb')
    data_names = pickle.load(data_names_file)
    ref_names_file = open(save_path+'ref-imagenames.pkl', 'rb')
    ref_names = pickle.load(ref_names_file)
    num_of_ref = len(ref_names)
    # 加载特征文件
    data_style_features_file = open(save_path+'data-style-features.pkl', 'rb')
    data_style_features = pickle.load(data_style_features_file)
    data_luminance = data_style_features['luminance_features']
    data_mu = data_style_features['color_mu']
    data_cov = data_style_features['color_cov']
    ref_style_features_file = open(save_path+'ref-style-features.pkl', 'rb')
    ref_style_features = pickle.load(ref_style_features_file)
    ref_luminance = ref_style_features['luminance_features']
    ref_mu = ref_style_features['color_mu']
    ref_cov = ref_style_features['color_cov']
    # 加载聚类文件
    cluster_index_file = open(save_path+'index.pkl', 'rb')
    cluster_index = pickle.load(cluster_index_file)
    num_of_clusters = len(cluster_index)
    score = np.zeros((num_of_clusters,num_of_ref))
    # 累积计分
    for i in range(num_of_clusters):
        c = cluster_index[i]
        for j in range(num_of_ref):
            for k in range(len(c)):
                scoring(i,j,c[k])
         
<<<<<<< HEAD
=======
    print(score)
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
    ranking_file = open(save_path+'style-ranking.pkl', 'wb')
    pickle.dump(score,ranking_file)


