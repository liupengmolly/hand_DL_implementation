# -*- coding:utf-8 -*-
import struct
import numpy as np
import pickle
from preprocess.config import cfg

def dump_param(parm,model_file):
    file = open(model_file,'wb')
    pickle.dump(parm,file)
    file.close()

def load_param(model_file):
    file = open(model_file,'rb')
    param = pickle.load(file)
    file.close()
    return param

def load_mnist(prefix,kind='train'):
    if kind == 'train':
        labels_path = prefix+"data/train-labels-idx1-ubyte"
        images_path = prefix+"data/train-images-idx3-ubyte"
    else:
        labels_path = prefix+"data/t10k-labels-idx1-ubyte"
        images_path = prefix+"data/t10k-images-idx3-ubyte"
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

if __name__ == '__main__':
    x_train,y_train = load_mnist()
    print(x_train.shape(),y_train.shape())

