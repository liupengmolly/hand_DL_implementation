# -*- coding:utf-8 -*-

import os
import numpy as np
class NN(object):
    def __init__(self,cfg):
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
                # self.Ws.append(np.ones((self.cfg.vector_length,self.cfg.units_num)))
            else:
                self.Ws.append(np.random.uniform(-1.0,1.0,(self.cfg.units_num, self.cfg.units_num)))
                # self.Ws.append(np.ones((self.cfg.units_num,self.cfg.units_num)))
            self.Hs.append(np.zeros((self.cfg.batch_size, self.cfg.units_num)))
            self.Os.append(np.zeros((self.cfg.batch_size, self.cfg.units_num)))
        self.Ws.append(np.random.uniform(-1.0, 1.0, (self.cfg.units_num,10)))
        self.Hs.append(np.zeros((self.cfg.batch_size, 10)))
        self.Os.append(np.zeros((self.cfg.batch_size, 10)))
        self.error = None
        self.act_func = self.sigmoid

    def sigmoid(self,h):
        return 1/(1+np.exp(-h))

    def softmax(self):
        tmp = self.Hs[self.cfg.layers_num]
        tmp = np.exp(tmp)
        tmp_sum = np.sum(tmp,1)
        tmp = tmp/np.expand_dims(tmp_sum,1)
        return tmp

    def forward(self,inputs):
        self.Hs[0] = np.matmul(inputs,self.Ws[0])
        self.Os[0] = self.sigmoid(self.Hs[0])
        for i in range(self.cfg.layers_num-1):
            self.Hs[i+1] = np.matmul(self.Os[i],self.Ws[i+1])
            self.Os[i+1] = self.sigmoid(self.Hs[i+1])
        self.Hs[self.cfg.layers_num] = np.matmul(self.Os[self.cfg.layers_num-1],self.Ws[self.cfg.layers_num])
        self.Os[self.cfg.layers_num] = self.softmax()

    def cross_entropy_loss(self,labels):
        self.one_hot_labels = np.eye(10)[labels]
        loss = np.sum(-np.sum(self.one_hot_labels*(np.log(self.Os[-1])),1))/self.cfg.batch_size
        return loss

    def backprop(self,inputs):
        last_derivation = self.Os[-1] - self.one_hot_labels
        self.Ws[-1] = self.Ws[-1] - self.cfg.lr * np.matmul(self.Os[-2].T, last_derivation)
        for i in range(self.cfg.layers_num-1,0,-1):
            last_derivation = np.matmul(last_derivation,
                                        np.matmul(self.Ws[i+1].T, np.matmul(self.Os[i].T,(1-self.Os[i]))))
            self.Ws[i] = self.Ws[i] - self.cfg.lr * np.matmul(self.Os[i-1].T, last_derivation)

        last_derivation = np.matmul(last_derivation,np.matmul(self.Ws[1].T,np.matmul(self.Os[0].T,(1-self.Os[0]))))
        self.Ws[0] = self.Ws[0] - self.cfg.lr*np.matmul(inputs.T, last_derivation)

    def predict(self,inputs):
        self.forward(inputs)
        return np.argmax(self.Os[-1],1)

