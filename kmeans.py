# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
# 加载特征数据
save_path = '/home/silence/proj/'
<<<<<<< HEAD
#file_feature = open(save_path+'features.pkl', 'rb') #训练
file_feature = open(save_path+'test_features.pkl', 'rb') #测试
features = pickle.load(file_feature)
### 训练
'''
# 聚类
num_clusters = 1000
=======
file_feature = open(save_path+'features.pkl', 'rb') #训练
#file_feature = open(save_path+'test_features.pkl', 'rb') #测试
features = pickle.load(file_feature)
### 训练

# 聚类
num_clusters = 3
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
km_cluster = KMeans(n_clusters=num_clusters,n_jobs=-1)
km_cluster.fit(features)
kmeans_model = open(save_path+'kmeans_model.pkl','wb')
pickle.dump(km_cluster,kmeans_model) #保存训练好的模型
file_cluster = open(save_path+'clusters.pkl', 'wb')
pickle.dump(km_cluster.labels_,file_cluster)
# 存储结果
labels = km_cluster.labels_.copy()
file_names = open(save_path+'photonames.pkl','rb')
photonames = np.array(pickle.load(file_names))
index_list = []
names_list = []
for label in range(num_clusters):
    index = np.where(labels == label)[0]
    index.sort()
    index_list.append(index.copy())
    names_list.append(photonames[index].copy())
file_index = open(save_path+'index.pkl','wb')
pickle.dump(index_list,file_index)
clustered_filenames = open(save_path+'clusteredNames.pkl','wb')
pickle.dump(names_list,clustered_filenames)
<<<<<<< HEAD
'''
### 测试

=======

### 测试
'''
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658
kmeans_model = open(save_path+'kmeans_model.pkl','rb')
km_cluster = pickle.load(kmeans_model)
labels = km_cluster.predict(features)
test_labels = open(save_path+'test_labels.pkl', 'wb')
pickle.dump(labels,test_labels)
<<<<<<< HEAD

=======
'''
>>>>>>> 4a6a0ca894d0fb8c2565e23efda9333522487658

