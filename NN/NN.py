# -*- coding:utf-8 -*-

import os
import numpy as np
class NN:
    def __init__(self,cfg):
        self.Ws = []
        for i in range(cfg.layers_num):
            if i==0:
                self.Ws.append(np.random.uniform(-1.0,1.0,(cfg.vector_length,cfg.units_num)))
            else:
                self.Ws.append(np.random.uniform(-1.0,1.0,(cfg.units_num,cfg.units_num)))
        self.Ws.append(np.random.uniform(-1.0,1.0,(cfg.units_num,10)))

    def forward(self):
        pass

    def b
