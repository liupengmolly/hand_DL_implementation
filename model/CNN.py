# -*- coding:utf-8 -*-
from model.util import *
import math

class CNN:
    def __init__(self, cfg, inputs_shape, conv_arg, pool_arg):
        """
        这个cnn适用于minist数据集，还不能单纯通过参数广泛适用

        最初版本：
        1）conv只支持输出相同size的卷积结果，stride为1
        2）使用max_pool, pool的大小和stride可以设定
        3）先只支持一层conv

        :param cfg:
        :param inputs_shape: (batch_size,height,width)
        :param conv_arg: (channels_num,height,width)
        :param pool_arg: (stride,height,width)
        """
        self.cfg = cfg
        self.inputs_shape = inputs_shape
        self.conv_arg = conv_arg
        self.pool_arg = pool_arg
        self.pad_height_len = int((conv_arg[1] - 1) / 2)
        self.pad_width_len = int((conv_arg[2] - 1) / 2)
        self.F = np.random.uniform(-1, 1, conv_arg)
        self.conv_layers = np.zeros((self.cfg.batch_size, conv_arg[0],
                                     inputs_shape[-2], inputs_shape[-1]))
        self.relu_layers = np.zeros((self.cfg.batch_size, conv_arg[0],
                                     inputs_shape[-2], inputs_shape[-2]))
        self.pool_layer_height = int((inputs_shape[-2] - pool_arg[1]) / pool_arg[0] + 1)
        self.pool_layer_width = int((inputs_shape[-1] - pool_arg[2]) / pool_arg[0] + 1)
        self.pool_layers = np.zeros((self.cfg.batch_size, conv_arg[0], self.pool_layer_height, self.pool_layer_width))
        self.pool_tags = np.zeros((self.cfg.batch_size, conv_arg[0], self.pool_layer_height, self.pool_layer_width, 2))
        flat_length = self.conv_arg[0] * self.pool_layer_height * self.pool_layer_width
        self.flat_layers = np.zeros((self.cfg.batch_size, flat_length))
        self.W = np.random.uniform(-1, 1, (flat_length, 10))
        self.H = np.zeros((self.cfg.batch_size, 10))
        self.O = np.zeros((self.cfg.batch_size, 10))

        self.init_lr = cfg.lr
        self.act_func = eval(cfg.act_func)
        self.act_deriv = eval(cfg.act_func + '_deriv')
        self.optimization = eval(cfg.optimization)

    def conv_cal(self, inputs, j, k, f):
        half_filter_height = math.ceil((f.shape[0] - 1) / 2)
        half_filter_width = math.ceil((f.shape[1] - 1) / 2)
        conv_matrix = inputs[:,
                             :,
                             j - half_filter_height:j - half_filter_height + f.shape[0],
                             k - half_filter_width:k - half_filter_width + f.shape[1]] \
                      * np.expand_dims(np.expand_dims(f, 0), 0)
        conv_result = np.sum(np.sum(np.sum(conv_matrix, 1), 1),1)
        return conv_result

    def get_conv_layers(self, inputs, F):
        if len(F.shape) == 4:
            F = np.sum(F,0) / F.shape[0]
        half_F_height = math.ceil((F.shape[-2]-1)/2)
        half_F_width = math.ceil((F.shape[-1]-1)/2)
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs,1)

        conv_layers = np.zeros((inputs.shape[0],F.shape[0],
                                inputs.shape[2]-F.shape[1]+1,inputs.shape[3]-F.shape[2]+1))
        for i,f in enumerate(F):
            for j in range(half_F_height, inputs.shape[-2] - half_F_height):
                for k in range(half_F_width, inputs.shape[-1] - half_F_width):
                    conv_layers[:, i, j-half_F_height, k-half_F_width] = self.conv_cal(inputs, j, k, f)
        return conv_layers

    def pool_cal(self, inputs, i, j):
        pool = inputs[:, :, i * self.pool_arg[0]: i * self.pool_arg[0] + self.pool_arg[1],
               j * self.pool_arg[0]: j * self.pool_arg[0] + self.pool_arg[2]]
        idx = pool.reshape(pool.shape[0], pool.shape[1], -1).argmax(-1)
        row, col = np.unravel_index(idx, pool.shape[-2:])
        pool_idx = np.stack((row, col), -1)
        return np.max(np.max(pool, 2), 2), pool_idx

    def get_pool_layers(self, inputs):
        pool_shape = self.pool_layers.shape
        for i in range(pool_shape[-2]):
            for j in range(pool_shape[-1]):
                self.pool_layers[:, :, i, j], self.pool_tags[:, :, i, j, :] = self.pool_cal(inputs, i, j)
        return self.pool_layers, self.pool_tags

    def recover_pool(self, delta_pool):
        recover_pool = np.zeros((self.cfg.batch_size * self.conv_arg[0],
                                 self.inputs_shape[1] * self.inputs_shape[2]))
        for i in range(delta_pool.shape[2]):
            for j in range(delta_pool.shape[3]):
                x_idx = i*self.pool_arg[1]+self.pool_tags[:,:,i,j,0].reshape(-1).astype(int)
                y_idx = j*self.pool_arg[2]+self.pool_tags[:,:,i,j,1].reshape(-1).astype(int)
                recover_pool[range(recover_pool.shape[0]), x_idx*28+y_idx] = \
                    delta_pool[:,:,i,j].reshape(-1)
        recover_pool = recover_pool.reshape(self.cfg.batch_size, self.conv_arg[0],
                                            self.inputs_shape[1], self.inputs_shape[2])
        return recover_pool

    def forward(self, inputs):
        self.pad_inputs = np.zeros((inputs.shape[0], inputs.shape[1] + 2, inputs.shape[2] + 2))
        for i in range(inputs.shape[0]):
            self.pad_inputs[i] = np.pad(inputs[i],
                                        ((self.pad_height_len, self.pad_height_len),
                                         (self.pad_width_len, self.pad_width_len)),
                                        'constant')
        self.conv_layers = self.get_conv_layers(self.pad_inputs, self.F)
        self.relu_layers = self.act_func(self.conv_layers)
        self.pool_layers, self.pool_tags = self.get_pool_layers(self.relu_layers)
        self.flat_layers = self.pool_layers.reshape((self.cfg.batch_size, -1))
        self.H = np.matmul(self.flat_layers, self.W)
        self.O = softmax(self.H)

    def backprop(self, inputs, labels):
        one_hot_labels = np.eye(10)[labels]
        last_derivation = self.O - one_hot_labels
        delta_W = np.matmul(self.flat_layers.T, last_derivation) / self.cfg.batch_size
        self.W = self.optimization(self.cfg.lr, self.W, delta_W)
        delta_flatten = np.matmul(last_derivation, self.W.T)
        delta_pool = delta_flatten.reshape(self.cfg.batch_size, self.conv_arg[0],
                                           self.pool_layer_height, self.pool_layer_width)
        recover_delta_pool = self.recover_pool(delta_pool)
        delta_relu = self.act_deriv(recover_delta_pool)
        delta_F = self.get_conv_layers(self.pad_inputs, delta_relu)
        delta_F = np.sum(delta_F, 0) / delta_F.shape[0]
        self.F = self.optimization(self.cfg.lr, self.F, delta_F)

    def predict(self, inputs):
        self.forward(inputs)
        return np.argmax(self.O, 1)
