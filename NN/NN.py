# -*- coding:utf-8 -*-

import os
import numpy as np
class NN:
    def __init__(self,cfg,inputs,labels):
        """
        初始化
        :param cfg:
        :param inputs: 输入是一个batch,这样每次模型计算或者更新时不用考虑迭代的情况
        :param labels:
        """
        self.cfg = cfg
        self.Ws = []
        self.Hs = []
        self.Os = []
        if self.cfg.layers_num < 1:
            return ValueError("the number of layers at least be 1,now is {}".format(self.cfg.layers_num))
        for i in range(self.cfg.layers_num):
            if i==0:
                self.Ws.append(np.random.uniform(-1.0,1.0,(self.cfg.vector_length, self.cfg.units_num)))
            else:
                self.Ws.append(np.random.uniform(-1.0,1.0,(self.cfg.units_num, self.cfg.units_num)))
            self.Hs.append(np.zeros((self.cfg.batch_size, self.cfg.units_num)))
            self.Os.append(np.zeros((self.cfg.batch_size, self.cfg.units_num)))
        self.Ws.append(np.random.uniform(-1.0, 1.0, (self.cfg.units_num,10)))
        self.Hs.append(np.zeros((self.cfg.batch_size, 10)))
        self.Os.append(np.zeros((self.cfg.batch_size, 10)))
        self.error = None
        self.inputs = np.array(inputs)
        self.labels = np.array(labels)
        self.act_func = self.sigmoid

    def sigmoid(self,O):
        return 1/(1+np.exp(-O))

    def softmax(self):
        tmp = self.Os[self.cfg.layers_num]
        tmp = np.exp(tmp)
        tmp_sum = np.sum(tmp,1)
        tmp = tmp/np.expand_dims(tmp_sum,1)
        return tmp

    def forward(self):
        self.Hs[0] = np.matmul(self.inputs,self.Ws[0].T)
        self.Os[0] = self.sigmoid(self.Hs[0])
        for i in range(self.cfg.layers_num-1):
            self.Hs[i+1] = np.matmul(self.Os[i],self.Ws[i+1].T)
            self.Os[i+1] = self.sigmoid(self.Hs[i+1])
        self.Os[self.cfg.layers_num] = np.matmul(self.Os[self.cfg.layers_num-1],self.Ws[self.cfg.layers_num].T)
        self.O = self.softmax()

    def cross_entropy_loss(self):
        one_hot_labesl = np.eye(10)[self.labels]
        loss = np.sum(-np.sum(one_hot_labesl*(np.log(self.O)),1))/self.cfg.batch_size
        return loss

    def backprop(self):


    def predict(self):
        self.forward()
        return np.argmax(self.O,1)

