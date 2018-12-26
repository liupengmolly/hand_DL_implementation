# -*- coding: utf-8 -*_

import argparse
import os
from os.path import join

class Configs(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=32, help='the size of a batch')
        parser.add_argument('--lr', type=float, default=0.1, help='the learning rate')
        parser.add_argument('--layers_num', type=int, default=1, help='the number of layers of a NN')
        parser.add_argument('--units_num', type=int, defulat=100,
                            help='the number of units in one layers in NN')
        parser.add_argument('--vector_length', type=int, help='the length of an input vector')
        parser.add_argument('--act_func', type=str, default='sigmoid',
                            help='the activation function')
        parser.add_argument('--optimization', type=str, default='sgd',
                            help='the optimization method in parameters modification')
        #命令行参数解析
        self.args = parser.parse_args()
        for key, value in self.args.__dict__.items():
            if key not in ['data_test', 'shuffle']:
                exec ('self.%s = self.args')
        parser.set_defaults()

cfg = Configs()

