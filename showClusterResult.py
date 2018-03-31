import os
import pickle
import numpy as np
from PIL import Image
from skimage import io, color,exposure,img_as_ubyte
import matplotlib.pyplot as plt
save_path = '/home/silence/proj/'
photo_path = '/home/silence/proj/photos/'
clustered_filenames = open(save_path+'clusteredNames.pkl','rb')
num_clusters = 1000
namelist = pickle.load(clustered_filenames)
for label in range(num_clusters):
    fig = plt.figure()
    for i in range(36):
        img = Image.open(photo_path+namelist[label][i])
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(img)
        plt.axis('off')
    plt.show()


