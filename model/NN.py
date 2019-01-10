# -*- coding:utf-8 -*-sigmoid

import numpy as np
from model.util import *

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
        self.init_lr = cfg.lr
        self.act_func = eval(cfg.act_func)
        self.act_deriv = eval(cfg.act_func+'_deriv')
        self.optimization = eval(cfg.optimization)


    def softmax(self):
        tmp = self.Hs[self.cfg.layers_num]
        tmp = np.exp(tmp-np.expand_dims(np.max(tmp,1),1))
        tmp_sum = np.sum(tmp,1)
        tmp = tmp/np.expand_dims(tmp_sum,1)
        return tmp

    def forward(self,inputs):
        self.Hs[0] = np.matmul(inputs,self.Ws[0])
        self.Os[0] = self.act_func(self.Hs[0])
        for i in range(self.cfg.layers_num-1):
            self.Hs[i+1] = np.matmul(self.Os[i],self.Ws[i+1])
            self.Os[i+1] = self.act_func(self.Hs[i+1])
        self.Hs[self.cfg.layers_num] = np.matmul(self.Os[self.cfg.layers_num-1],
                                                 self.Ws[self.cfg.layers_num])
        self.Os[self.cfg.layers_num] = self.softmax()

    def cross_entropy_loss(self,labels):
        one_hot_labels = np.eye(10)[labels]
        loss = np.sum(-np.sum(one_hot_labels*(np.log(self.Os[-1])),1))/self.cfg.batch_size
        return loss

    def backprop(self,inputs,labels):
        """
        注意：
        1 激活函数的求导是element wise
        2 由于向量内积导致一个batch内的值加和需要除去batch_size
        :return:
        """
        one_hot_labels = np.eye(10)[labels]
        last_derivation = self.Os[-1] - one_hot_labels
        derivation = np.matmul(self.Os[-2].T, last_derivation)/self.cfg.batch_size
        self.Ws[-1] = self.optimization(self.cfg.lr,self.Ws[-1],derivation)
        for i in range(self.cfg.layers_num-1,0,-1):
            last_derivation = np.matmul(last_derivation,self.Ws[i+1].T)*self.act_deriv(self.Os[i])
            derivation = np.matmul(self.Os[i-1].T, last_derivation)/self.cfg.batch_size
            self.Ws[i] = self.optimization(self.cfg.lr,self.Ws[i],derivation)

        last_derivation = np.matmul(last_derivation, self.Ws[1].T)*self.act_deriv(self.Os[0])
        derivation = np.matmul(inputs.T, last_derivation)/self.cfg.batch_size
        self.Ws[0] = self.optimization(self.cfg.lr,self.Ws[0],derivation)

    def predict(self,inputs):
        self.forward(inputs)
        return np.argmax(self.Os[-1],1)

    def line_decay_lr(self,k):
        if k<=100:
            self.cfg.lr = (1-k/100)*self.init_lr + (k/100)*0.01*self.init_lr


