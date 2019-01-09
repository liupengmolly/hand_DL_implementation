# -*- coding:utf-8 -*-

from preprocess.read import *
from preprocess.config import cfg
from NN.NN import NN
from itertools import *
import random

def get_batches(data,round = 1,shuffle = True):
    for _ in range(round):
        if shuffle:
            random.shuffle(data)
        cur_index,next_index = None,None
        for batch_index in range(int(len(data)/cfg.batch_size)):
            cur_index = cfg.batch_size * batch_index
            next_index = cfg.batch_size * (batch_index+1)
            yield data[cur_index:next_index]
        yield data[next_index:]+[(np.zeros(cfg.vector_length),0) for _ in range(len(data)-next_index)]

def cal_ac(nn,test):
    predicts = []
    x_test,y_test = zip(*test)
    x_test,y_test = np.array(x_test),np.array(y_test)
    for batch in get_batches(test,shuffle=False):
        x_batch,y_batch = zip(*batch)
        x_batch,y_batch = np.array(x_batch),np.array(y_batch)
        predicts.append(nn.predict(x_batch))
    #get_batches得到的最后一个batch因为补全可能会重复用到前面的数据
    predicts = np.concatenate(tuple(predicts))[:len(test)]
    ac = np.sum(predicts==y_test)/len(test)
    return ac

images, labels = load_mnist()
train_data = list(zip(list(images),list(labels)))

nn = NN(cfg)
loss = 100
for batch in get_batches(train_data,10):
    if loss<0.1:
        break
    x_train,y_train = zip(*batch)
    x_train,y_train = np.array(x_train),np.array(y_train)
    nn.forward(x_train)
    loss = nn.cross_entropy_loss(y_train)
    print(loss)
    nn.backprop(x_train)

x_test,y_test = load_mnist('test')
test_data = list(zip(list(x_test),list(y_test)))
print(cal_ac(nn,test_data))
