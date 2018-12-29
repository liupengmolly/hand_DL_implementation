# -*- coding:utf-8 -*-
import os
import struct
import numpy as np

def load_mnist(kind='train'):
    if kind == 'train':
        labels_path = "../data/train-labels-idx1-ubyte"
        images_path = "../data/train-images-idx3-ubyte"
    else:
        labels_path = "../data/t10k-labels-idx1-ubyte"
        images_path = "../data/t10k-images-idx3-ubyte"
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

