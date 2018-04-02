import os
import pickle
import numpy as np
from PIL import Image
from skimage import io, color,exposure,img_as_ubyte
import matplotlib.pyplot as plt
save_path = '/home/silence/proj/'
photo_path = '/home/silence/proj/photos/'
'''
clustered_filenames = open(save_path+'clusteredNames.pkl','rb')
names_list = pickle.load(clustered_filenames)
print(names_list)
'''
'''
image_names_file = open(save_path+'refimagenames.pkl', 'rb')
image_names = pickle.load(image_names_file)
print(image_names)
'''
'''
style_features = open(save_path+'data-style-features.pkl', 'rb')
features = pickle.load(style_features)
print(features)
'''
'''
style_ranking = open(save_path+'style-ranking.pkl', 'rb')
ranking = pickle.load(style_ranking)
print(ranking)
'''

'''
dir_path = '/home/silence/proj/photos_demo/'
image_list = os.listdir(dir_path)
image_list.sort()
image_list2 = os.listdir(dir_path)
l = list()
for image in sorted(image_list2):
    l.append(image)
print(image_list)
print(l)
'''
'''
test_names = open(save_path+'test_photonames.pkl', 'rb')
tnames = pickle.load(test_names)
print(tnames)
'''
'''
test_labels = open(save_path+'test_labels.pkl', 'rb')
labels = pickle.load(test_labels)
print(labels)
'''
'''
clustered_filenames = open(save_path+'clusteredNames.pkl','rb')
namelist = pickle.load(clustered_filenames)
for name in namelist[2]:
    #ima = io.imread(photo_path+name)
    #plt.imshow(ima)
    #plt.axis('off') # 不显示坐标轴
    #plt.show()

    img = Image.open(photo_path+name)  
    fig = plt.figure()  
    ax = fig.add_subplot(121)  
    ax.imshow(img)  
    plt.axis('off')
    ax = fig.add_subplot(122)  
    ax.imshow(img)#以灰度图显示图片  
    plt.axis('off')
    plt.show()
'''
