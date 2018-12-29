# -*- coding: utf-8 -*_

import argparse
import os
from os.path import join

class Configs(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=64, help='the size of a batch')
        parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
        parser.add_argument('--layers_num', type=int, default=3, help='the number of layers of a NN')
        parser.add_argument('--units_num', type=int, default=100,
                            help='the number of units in one layers in NN')
        parser.add_argument('--vector_length', type=int, default=784,help='the length of an input vector')
        parser.add_argument('--act_func', type=str, default='sigmoid',
                            help='the activation function')
        parser.add_argument('--optimization', type=str, default='sgd',
                            help='the optimization method in parameters modification')
        #命令行参数解析
        self.args = parser.parse_args()
        for key, value in self.args.__dict__.items():
            exec ('self.%s = self.args.%s'%(key,key))
        parser.set_defaults()

cfg = Configs()

