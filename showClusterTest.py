import os
import pickle
import numpy as np
from PIL import Image
from skimage import io, color,exposure,img_as_ubyte
import matplotlib.pyplot as plt
save_path = '/home/silence/proj/'
photo_path = '/home/silence/proj/photos/'
test_path = '/home/silence/proj/photos_test/'
clustered_filenames = open(save_path+'clusteredNames.pkl','rb')
namelist = pickle.load(clustered_filenames)
test_labels = open(save_path+'test_labels.pkl','rb')
labels = pickle.load(test_labels)
file_names = open(save_path+'test_photonames.pkl','rb')
names = pickle.load(file_names)
photo_list = os.listdir(test_path)
i = 0
for photo in sorted(photo_list):
    label = labels[i]
    fig = plt.figure()
    im = Image.open(test_path+photo)
    ax = fig.add_subplot(3,4,1)
    ax.imshow(im)
    ax.set_title('Input image')
    plt.axis('off')
    d = 0
    for j in range(9):
        img = Image.open(photo_path+namelist[label][j])
        ax = fig.add_subplot(3,4,j+1+int(d/3)+1)
        ax.imshow(img)
        plt.axis('off')
        if j==1:
             ax.set_title('Cluster')
        d += 1
    plt.show()
    i += 1

