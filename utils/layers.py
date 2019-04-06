import numpy as np
import math
from utils.act_func import *
from utils.opt import *

class Conv_2d:
    def __init__(self, cfg, inputs_shape, conv_arg):
        """
        卷积层, 卷积移动的方式暂时只支持‘valid’模式, 且卷积计算的stride都默认为1

        :param cfg: 配置参数
        :param inputs_shape: 输入数据的相关数据，(batch_size,height,width,'maybe the channel')
        :param conv_arg:  卷积层的相关参数, (height, width, input_channel, output_channel)
        """
        self.cfg = cfg
        self.half_F = int((conv_arg[0]-1)/2)
        self.conv_arg = conv_arg
        self.F = np.random.uniform(-1, 1, conv_arg)
        self.layers = np.zeros((inputs_shape[0] ,
                                inputs_shape[1]-2*self.half_F,
                                inputs_shape[2]-2*self.half_F,
                                conv_arg[-1]))
        self.optimization = eval(cfg.optimization+'()')

    def conv_cal(self, inputs, F):
        """
        innuts和F都是经过调整，维度一致的，卷积计算默认stride都等于1

       :param inputs: (batch_size, height, width, channel1, channel2), 其中channel1一般等于1
        :param F: (1, kernel_height, kernel_width, channel2, channel3)
        :return:
        """
        layers = np.zeros((inputs.shape[0],
                           inputs.shape[1]-F.shape[1]+1,
                           inputs.shape[2]-F.shape[2]+1,
                           inputs.shape[-2],
                           F.shape[4]))
        for i in range(layers.shape[1]):
            for j in range(layers.shape[2]):
                layers[:, i, j, :, :] = np.sum(np.sum(np.matmul(
                    inputs[:,i:i+F.shape[1],j:j+F.shape[2], :, :], F[:, :, :, :, :]
                ), 1), 1)
        return layers

    def forward(self, inputs):
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, -1)

        #注意下面的inputs和F都是临时变量
        inputs = np.expand_dims(inputs,-2)
        F = np.expand_dims(self.F, 0)

        tmp_layers = self.conv_cal(inputs, F)
        self.layers = np.sum(tmp_layers, -2)
        return self.layers

    def backprop(self, inputs, lr, delta, deriv_layer=False):
        """
        反向传播
        :param lr: 学习率
        :param delta: 对上一层的偏导值，（batch_size, height, width, out_channel)
        :param deriv_layer: 是否对下一层求导（意味着从反向角度考虑是否有下一层）
        :return:
        """
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, -1)
        inputs = np.expand_dims(inputs, -1)
        delta_for_F = np.expand_dims(delta, -2)
        delta_F = self.conv_cal(inputs, delta_for_F)
        delta_F = np.sum(delta_F, 0) / delta_F.shape[0]

        self.F1 = self.optimization.optimize(self.F, lr, delta_F)
        delta_pool = None

        if deriv_layer:
            pad_delta = np.pad(delta,(
                (0, 0),
                (self.F.shape[0]-1, self.F.shape[0]-1),
                (self.F.shape[1]-1, self.F.shape[1]-1),
                (0, 0)), 'constant')
            pad_delta = np.expand_dims(pad_delta, -2)

            F_for_pool = np.expand_dims(np.rot90(self.F, 2, (0,1)).swapaxes(2,3), 0)
            delta_pool = self.conv_cal(pad_delta, F_for_pool)
            delta_pool = np.sum(delta_pool, -2)
        return delta_pool

class MaxPool:
    def __init__(self, cfg, inputs_shape, pool_arg):
        """
        最大池化层
        :param cfg:
        :param inputs: 上一层激活曾或者卷积层的输出，（batch_size, height, width, channel)
        :param pool_arg: (stride, height, width)
        """
        self.cfg = cfg
        self.inputs_shape = inputs_shape
        self.pool_arg = pool_arg
        self.pool_layer_height = int((inputs_shape[1] - pool_arg[1])/pool_arg[0])+1
        self.pool_layer_width = int((inputs_shape[2] - pool_arg[2])/pool_arg[0])+1
        self.pool_layer = np.zeros((self.cfg.batch_size, self.pool_layer_height,
                                    self.pool_layer_width, inputs_shape[3]))

    def pool_cal(self, inputs, i, j, pool_arg):
        pool = inputs[:, i * pool_arg[0]: i * pool_arg[0] + pool_arg[1],
               j * pool_arg[0]: j * pool_arg[0] + pool_arg[2], :]
        idx = pool.reshape(pool.shape[0], -1, pool.shape[-1])
        idx = idx.argmax(-2)
        row, col = np.unravel_index(idx, pool.shape[1:-1])
        pool_idx = np.stack((row, col), -1)
        return np.max(np.max(pool, 1), 1), pool_idx

    def forward(self, inputs):
        """

        :param inputs: (batch_size, height, width, channel_num)
        :return:
        """
        pool_tags = np.zeros((self.cfg.batch_size, self.pool_layer_height,
                              self.pool_layer_width, self.inputs_shape[3], 2))
        for i in range(self.pool_layer_height):
            for j in range(self.pool_layer_width):
                self.pool_layer[:, i, j, :], pool_tags[:, i, j, :, :] \
                    = self.pool_cal(inputs, i, j, self.pool_arg)
        return self.pool_layer, pool_tags

    def recover_pool(self, delta, pool_arg, pool_tag):
        recover_pool = np.zeros((self.cfg.batch_size * delta.shape[-1],
                                 delta.shape[1] * pool_arg[1] *delta.shape[2] * pool_arg[2]))
        for i in range(delta.shape[1]):
            for j in range(delta.shape[2]):
                x_idx = i*pool_arg[1]+pool_tag[:,i,j,:,0].reshape(-1).astype(int)
                y_idx = j*pool_arg[2]+pool_tag[:,i,j,:,1].reshape(-1).astype(int)
                recover_pool[range(recover_pool.shape[0]), x_idx*delta.shape[2]*pool_arg[2]+y_idx] = \
                    delta[:,i,j,:].reshape(-1)
        recover_pool = recover_pool.reshape(self.cfg.batch_size, delta.shape[-1],
                                            delta.shape[1] * pool_arg[1] *delta.shape[2] * pool_arg[2])
        recover_pool = np.swapaxes(recover_pool,1,2)
        recover_pool = recover_pool.reshape(self.cfg.batch_size, delta.shape[1] * pool_arg[1],
                                            delta.shape[2] * pool_arg[2], delta.shape[-1])
        return recover_pool

    def backprop(self, delta, pool_tags):
        delta_pool = self.recover_pool(delta, self.pool_arg, pool_tags)
        return delta_pool

class Batch_Normalization:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W = np.random.uniform(-1.0, 1.0,(self.cfg.units_num, self.cfg.units_num))
        self.b = np.random.uniform(-1.0, 1.0,(self.cfg.batch_size, self.cfg.units_num))
        self.delta = 1e-8
        self.mean_inptus = None
        self.std_inputs = None
        self.optimization = eval(self.cfg.optimization+"()")

    def forward(self, inputs):
        self.mean_inputs = np.expand_dims(np.sum(inputs,-1)/inputs.shape[-1],-1)
        self.std_inputs = np.sqrt(self.delta+np.expand_dims(
            np.sum(np.power(inputs-self.mean_inputs,2),-1)/inputs.shape[-1],-1))
        self.H = (inputs-self.mean_inputs)/self.std_inputs
        linear_H = np.matmul(self.H, self.W)+self.b
        return linear_H

    def backprop(self, lr, derivation):
        foredelta = np.matmul(derivation,self.W)/self.std_inputs
        self.b = self.optimization.optimize(self.b, lr, derivation,1)
        derivation = np.matmul(self.H.T, derivation)/self.cfg.batch_size
        self.W = self.optimization.optimize(self.W, lr, derivation,0)
        return foredelta
