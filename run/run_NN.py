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
while loss>0.05:
    if loss<0.5:
        nn.cfg.lr=0.05
    if loss<0.3:
        nn.cfg.lr = 0.03
    if loss<0.1:
        nn.cfg.lr = 0.01
    nn.forward()
    loss = nn.cross_entropy_loss()
    nn.backprop()
    i = i+1
    start_index = (i*64)%5936
    if i==100:
        print(i)
    nn.inputs = images[start_index:start_index+64]
    nn.labels = labels[start_index:start_index+64]

x_test,y_test = load_mnist('test')
nn.inputs = x_test[:64]
print(nn.predict())
print(y_test[:64])
