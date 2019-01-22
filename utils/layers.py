import numpy as np
import math
from utils.util import *

class Conv_2d:
    def __init__(self, cfg, inputs, conv_arg):
        """
        卷积层, 卷积移动的方式暂时只支持‘valid’模式

        :param cfg: 配置参数
        :param inputs_shape: 输入数据的相关数据，(batch_size,height,width)
        :param conv_arg:  卷积层的相关参数, (height, width, input_channel, output_channel)
        """
        if conv_arg[0] != conv_arg[1]:
            yield ValueError("the filter height must be same with the width, while the heights is"
                              " {} and the width is {}".format(conv_arg[0], conv_arg[1]))
        self.cfg = cfg
        self.half_F = int((conv_arg[0]-1)/2)
        self.inputs = inputs
        self.conv_arg = conv_arg
        self.F = np.random.uniform(-1, 1, conv_arg)
        self.layers = np.zeros((inputs.shape[0] ,
                                inputs.shape[-2]-2*self.half_F,
                                inputs.shape[-1]-2*self.half_F,
                                conv_arg[-1]))
        self.optimization = eval(cfg.optimization)

    def get_conv_layers(self, inputs, F):
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs,-1)
        if len(F.shape) == 3:
            F = np.expand_dims(F, -2)

        inputs = np.expand_dims(inputs, -2)
        F = np.expand_dims(F, 0)

        return np.sum(np.matmul(inputs,F),-2)

    def forward(self):
        if len(self.inputs.shape) == 3:
            self.inputs = np.expand_dims(self.inputs, -1)

        #注意下面的inputs和F都是临时变量
        inputs = np.expand_dims(self.inputs,-2)
        F = np.expand_dims(self.F, 0)

        return np.sum(np.matmul(inputs, F), -2)

    def backprop(self,lr, delta, deriv_layer=False):
        """
        反向传播
        :param lr: 学习率
        :param delta: 对上一层的偏导值，（batch_size, height, width, out_channel)
        :param deriv_layer: 是否对下一层求导（意味着从反向角度考虑是否有下一层）
        :return:
        """
        inputs = np.expand_dims(self.inputs, -2)
        delta_for_F = np.expand_dims(delta, )
        delta_F = self.get_conv_layers(self.inputs, delta)
        delta_F = np.sum(delta_F, 0) / delta_F.shape[0]
        self.F2 = self.optimization(lr, self.F, delta_F)

        if deriv_layer:
            delta = delta.reshape(-1, delta.shape[2], delta.shape[3])
            pad_delta = np.zeros((delta.shape[0],
                                  delta.shape[1]+2*(self.F.shape[1]-1),
                                  delta.shape[2]+2*(self.F.shape[2]-1)))
            for i in range(delta.shape[0]):
                pad_delta[i] = np.pad(delta[i],((self.F.shape[1]-1, self.F.shape[1]-1),
                                                (self.F.shape[2]-1, self.F.shape[2]-1)))
            pad_delta = pad_delta.reshape(self.cfg.batch_size, self.conv_arg[0],
                                          pad_delta.shape[1], pad_delta.shape[2])
            delta_pool = self.get_conv_layers(pad_delta, np.rot(self.F, 2, (1,2)))
            delta_pool = np.sum(delta_pool)
        delta_layer = self.

