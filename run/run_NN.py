# -*- coding:utf-8 -*-

from preprocess.read import *
from preprocess.config import cfg
from NN.NN import NN

images, labels = load_mnist()
print(images.shape,labels.shape)
print(images[0].shape,labels[0].shape)
print(images[0],labels[0])

x_train, y_train = images[:64],labels[:64]
nn = NN(cfg,x_train,y_train)
loss = 100
i = 0
while loss>1:
    nn.forward()
    loss = nn.cross_entropy_loss()
    nn.backprop()
    i = i+1
    start_index = (i*64)
    nn.inputs = images[start_index:start_index+64]
    nn.labels = labels[start_index:start_index+64]
    print(nn.predict())

x_test,y_test = load_mnist('test')
nn.inputs = x_test[:64]
print(nn.predict())
print(y_test[:64])
