import os
import pickle
import numpy as np
from PIL import Image
from skimage import io, color,exposure,img_as_ubyte
import matplotlib.pyplot as plt
top_k = 3
save_path = '/home/silence/proj/'
photo_path = '/home/silence/proj/photos/'
test_path = '/home/silence/proj/photos_test/'
ref_path = '/home/silence/proj/style_ref/'
clustered_filenames = open(save_path+'clusteredNames.pkl','rb')
namelist = pickle.load(clustered_filenames)
test_labels = open(save_path+'test_labels.pkl','rb')
labels = pickle.load(test_labels)
file_names = open(save_path+'test_photonames.pkl','rb')
names = pickle.load(file_names)
image_names_file = open(save_path+'ref-imagenames.pkl', 'rb')
ref_imagenames = np.array(pickle.load(image_names_file))
ranking_file = open(save_path+'style-ranking.pkl', 'rb')
style_ranking_all = pickle.load(ranking_file)
photo_list = os.listdir(test_path)
i = 0
for photo in sorted(photo_list):
    label = labels[i]
    fig = plt.figure()
    im = Image.open(test_path+photo)
    ax = fig.add_subplot(1,4,1)
    ax.imshow(im)
    ax.set_title('Input image')
    plt.axis('off')
    style_ranking = style_ranking_all[label].copy()
    top_styles = style_ranking.argsort()[::-1][:top_k] 
    ref_images = ref_imagenames[top_styles].copy()
    j = 2
    for ref_image in ref_images:
        ref_im = Image.open(ref_path+ref_image)
        ax = fig.add_subplot(1,4,j)
        ax.imshow(ref_im)
        plt.axis('off')
        if j==3:
             ax.set_title('Style references')
        j += 1
    plt.show()
    i += 1

