# -*- coding:utf-8 -*-

from preprocess.read import *
from preprocess.config import cfg
from NN.NN import NN
from itertools import *
import random

def get_batches(data,round = 1):
    batch_index = 0
    # perm_data = permutations(data)
    # data = perm_data.__next__()
    for _ in range(round):
        # while batch_index+cfg.batch_size <= len(data):
        random.shuffle(data)
        for batch_index in range(int(len(data)/cfg.batch_size)):
            yield data[batch_index*cfg.batch_size:batch_index*cfg.batch_size+cfg.batch_size]

def cal_ac(nn,x_test,y_test):
    predicts = []
    for batch in get_batches(x_test):
        predicts.append(nn.predict(batch))
    #get_batches得到的最后一个batch因为补全可能会重复用到前面的数据
    predicts = np.concatenate(tuple(predicts))[:len(y_test)]
    ac = np.sum(predicts==y_test)/len(y_test)
    return ac

images, labels = load_mnist()
train_data = list(zip(list(images),list(labels)))

nn = NN(cfg)
loss = 100
for batch in get_batches(train_data,1):
    if loss<0.1:
        break
    x_train,y_train = zip(*batch)
    x_train,y_train = np.array(x_train),np.array(y_train)
    nn.forward(x_train)
    loss = nn.cross_entropy_loss(y_train)
    print(loss)
    nn.backprop(x_train)

x_test,y_test = load_mnist('test')
print(cal_ac(nn,x_test,y_test))
