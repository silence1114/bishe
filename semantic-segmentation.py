import numpy as np
from PIL import Image
import sys
import vis
# 设置当前的工作环境在caffe下
caffe_root = '/home/silence/caffe/caffe/'
# 把caffe/python也添加到当前环境
sys.path.insert(0, caffe_root + 'python')
import caffe
dir_path = '/home/silence/proj/seg-test/'
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open(dir_path+'test3.jpeg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))
# load net
net = caffe.Net('/home/silence/proj/'+'voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
voc_palette = vis.make_palette(21)
out_im = Image.fromarray(vis.color_seg(out, voc_palette))
out_im.save(dir_path+'mask3.png')
