# -*- coding:utf-8 -*-sigmoid

from utils.util import *

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
        self.mu = []
        for i in range(self.cfg.layers_num):
            if i==0:
                self.Ws.append(np.random.uniform(-1.0, 1.0,(self.cfg.vector_length, self.cfg.units_num)))
                self.mu.append(np.zeros((self.cfg.vector_length, self.cfg.units_num)))
                # self.Ws.append(np.ones((self.cfg.vector_length,self.cfg.units_num)))
            else:
                self.Ws.append(np.random.uniform(-1.0, 1.0, (self.cfg.units_num, self.cfg.units_num)))
                self.mu.append(np.zeros((self.cfg.units_num, self.cfg.units_num)))
                # self.Ws.append(np.ones((self.cfg.units_num,self.cfg.units_num)))
            self.Hs.append(np.zeros((self.cfg.batch_size, self.cfg.units_num)))
            self.Os.append(np.zeros((self.cfg.batch_size, self.cfg.units_num)))
        self.Ws.append(np.random.uniform(-1.0, 1.0, (self.cfg.units_num, 10)))
        self.mu.append(np.zeros((self.cfg.units_num, 10)))

        self.Hs.append(np.zeros((self.cfg.batch_size, 10)))
        self.Os.append(np.zeros((self.cfg.batch_size, 10)))
        self.init_lr = cfg.lr
        self.act_func = eval(cfg.act_func)
        self.act_deriv = eval(cfg.act_func+'_deriv')
        self.optimization = eval(cfg.optimization)

    def forward(self, inputs):
        self.Hs[0] = np.matmul(inputs, self.Ws[0])
        self.Os[0] = self.act_func(self.Hs[0])
        for i in range(self.cfg.layers_num-1):
            self.Hs[i+1] = np.matmul(self.Os[i], self.Ws[i+1])
            self.Os[i+1] = self.act_func(self.Hs[i+1])
        self.Hs[self.cfg.layers_num] = np.matmul(self.Os[self.cfg.layers_num-1],
                                                 self.Ws[self.cfg.layers_num])
        self.Os[self.cfg.layers_num] = softmax(self.Hs[self.cfg.layers_num])

    def backprop(self, inputs, labels):
        """
        注意：
        1 激活函数的求导是element wise
        2 由于向量内积导致一个batch内的值加和需要除去batch_size
        :return:
        """
        one_hot_labels = np.eye(10)[labels]
        last_derivation = self.Os[-1] - one_hot_labels
        derivation = np.matmul(self.Os[-2].T, last_derivation)/self.cfg.batch_size
        self.Ws[-1] = self.optimization(self.cfg.lr, self.Ws[-1], derivation)
        self.mu[-1] = np.exp(-np.abs(derivation)/np.abs(self.Ws[-1] + derivation))
        for i in range(self.cfg.layers_num-1, 0, -1):
            last_derivation = np.matmul(last_derivation, self.Ws[i+1].T)*self.act_deriv(self.Os[i])
            derivation = np.matmul(self.Os[i-1].T, last_derivation)/self.cfg.batch_size
            self.Ws[i] = self.optimization(self.cfg.lr, self.Ws[i], derivation)
            self.mu[i] = np.exp(-np.abs(derivation)/np.abs(self.Ws[i] + derivation))

        last_derivation = np.matmul(last_derivation, self.Ws[1].T)*self.act_deriv(self.Os[0])
        derivation = np.matmul(inputs.T, last_derivation) / self.cfg.batch_size
        self.Ws[0] = self.optimization(self.cfg.lr, self.Ws[0], derivation)
        self.mu[0] = np.exp(-np.abs(derivation)/np.abs(self.Ws[0] + derivation))

    def predict(self, inputs):
        self.forward(inputs)
        return np.argmax(self.Os[-1], 1)
