# -*- coding:utf-8 -*-
from model.util import *
import math

class CNN:
    def __init__(self, cfg, inputs_shape, conv_arg1, pool_arg1, conv_arg2 = None, pool_arg2 = None):
        """
        这个cnn适用于minist数据集，还不能单纯通过参数广泛适用

        最初版本：
        1）conv只支持valid的卷积结果(第一层靠对初始数据的填充实现same)，stride为1
        2）使用max_pool, pool的大小和stride可以设定
        3）先只支持一层conv

        :param cfg:
        :param inputs_shape: (batch_size,height,width)
        :param conv_arg: (channels_num,height,width)
        :param pool_arg: (stride,height,width)
        """

        self.cfg = cfg
        self.inputs_shape = inputs_shape
        self.conv_arg1, self.conv_arg2, self.pool_arg1, self.pool_arg2 = \
            conv_arg1, conv_arg2, pool_arg1, pool_arg2
        self.half_f1, self.half_f2 = int((conv_arg1[1]-1)/2), int((conv_arg2[1]-1)/2)

        self.F1 = np.random.uniform(-1, 1, conv_arg1)
        self.conv_layer1 = np.zeros((self.cfg.batch_size, conv_arg1[0],
                                     inputs_shape[-2]-2*self.half_f1, inputs_shape[-1]-2*self.half_f1))
        self.relu_layer1 = np.zeros(self.conv_layer1.shape)
        self.pool_layer1_height = int((self.relu_layer1.shape[2] - pool_arg1[1]) / pool_arg1[0] + 1)
        self.pool_layer1_width = int((self.relu_layer1.shape[3] - pool_arg1[2]) / pool_arg1[0] + 1)
        self.pool_layer1 = np.zeros((self.cfg.batch_size, conv_arg1[0],
                                     self.pool_layer1_height, self.pool_layer1_width))
        self.pool_tags1 = np.zeros((self.cfg.batch_size, conv_arg1[0],
                                    self.pool_layer1_height, self.pool_layer1_width, 2))

        self.F2 = np.random.uniform(-1, 1, conv_arg2)
        self.conv_layer2 = np.zeros((self.cfg.batch_size, conv_arg2[0],
                                     self.pool_layer1.shape[2]-2*self.half_f2,
                                     self.pool_layer1.shape[2]-2*self.half_f2))
        self.relu_layer2 = np.zeros(self.conv_layer2.shape)
        self.pool_layer2_height = int((self.relu_layer2.shape[2] - pool_arg2[1]) / pool_arg2[0] + 1)
        self.pool_layer2_width = int((self.relu_layer2.shape[3] - pool_arg2[2]) / pool_arg2[0] + 1)
        self.pool_layer2 = np.zeros((self.cfg.batch_size, conv_arg2[0], self.pool_layer2_height, self.pool_layer2_width))
        self.pool_tags2 = np.zeros((self.cfg.batch_size, conv_arg2[0], self.pool_layer2_height, self.pool_layer2_width, 2))

        flat_length = self.conv_layer2.shape[1] * self.pool_layer2_height * self.pool_layer2_width

        self.flat_layers = np.zeros((self.cfg.batch_size, flat_length))
        self.W = np.random.uniform(-1, 1, (flat_length, 10))
        self.H = np.zeros((self.cfg.batch_size, 10))
        self.O = np.zeros((self.cfg.batch_size, 10))

        self.init_lr = cfg.lr
        self.act_func = eval(cfg.act_func)
        self.act_deriv = eval(cfg.act_func + '_deriv')
        self.optimization = eval(cfg.optimization)

    def conv_cal(self, inputs, j, k, F):
        half_filter_height = math.ceil((F.shape[1] - 1) / 2)
        half_filter_width = math.ceil((F.shape[2] - 1) / 2)
        inputs = np.expand_dims(inputs,1)
        conv_matrix = inputs[:,
                             :,
                             :,
                             j - half_filter_height:j - half_filter_height + F.shape[1],
                             k - half_filter_width:k - half_filter_width + F.shape[2]] \
                      * np.expand_dims(np.expand_dims(F,1), 0)
        conv_result = np.sum(np.sum(np.sum(conv_matrix, 2), 2),2)
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

        for j in range(half_F_height, inputs.shape[-2] - half_F_height):
            for k in range(half_F_width, inputs.shape[-1] - half_F_width):
                conv_layers[:, :, j-half_F_height, k-half_F_width] = self.conv_cal(inputs, j, k, F)
        return conv_layers

    def pool_cal(self, inputs, i, j, pool_arg):
        pool = inputs[:, :, i * pool_arg[0]: i * pool_arg[0] + pool_arg[1],
               j * pool_arg[0]: j * pool_arg[0] + pool_arg[2]]
        idx = pool.reshape(pool.shape[0], pool.shape[1], -1)
        idx = idx.argmax(-1)
        row, col = np.unravel_index(idx, pool.shape[-2:])
        pool_idx = np.stack((row, col), -1)
        return np.max(np.max(pool, 2), 2), pool_idx

    def get_pool_layers(self, inputs, pool_arg, pool_layer,pool_tag):
        pool_shape = pool_layer.shape
        for i in range(pool_shape[-2]):
            for j in range(pool_shape[-1]):
                pool_layer[:, :, i, j], pool_tag[:, :, i, j, :] \
                    = self.pool_cal(inputs, i, j, pool_arg)
        return pool_layer, pool_tag

    def recover_pool(self, delta_pool, pool_arg, pool_tag):
        recover_pool = np.zeros((self.cfg.batch_size * delta_pool.shape[1],
                                 delta_pool.shape[2] * pool_arg[1] *delta_pool.shape[3] * pool_arg[2]))
        for i in range(delta_pool.shape[2]):
            for j in range(delta_pool.shape[3]):
                x_idx = i*pool_arg[1]+pool_tag[:,:,i,j,0].reshape(-1).astype(int)
                y_idx = j*pool_arg[2]+pool_tag[:,:,i,j,1].reshape(-1).astype(int)
                recover_pool[range(recover_pool.shape[0]), x_idx*delta_pool.shape[3]*pool_arg[2]+y_idx] = \
                    delta_pool[:,:,i,j].reshape(-1)
        recover_pool = recover_pool.reshape(self.cfg.batch_size, delta_pool.shape[1],
                                            delta_pool.shape[2] * pool_arg[1],
                                            delta_pool.shape[3] * pool_arg[2])
        return recover_pool

    def forward(self, pad_inputs):
        self.conv_layer1 = self.get_conv_layers(pad_inputs, self.F1)
        self.relu_layer1 = self.act_func(self.conv_layer1)
        self.pool_layer1, self.pool_tags1 = self.get_pool_layers(self.relu_layer1, self.pool_arg1,
                                                                 self.pool_layer1, self.pool_tags1)

        self.conv_layer2 = self.get_conv_layers(self.pool_layer1, self.F2)
        self.relu_layer2 = self.act_func(self.conv_layer2)
        self.pool_layer2, self.pool_tags2 = self.get_pool_layers(self.relu_layer2, self.pool_arg2,
                                                                 self.pool_layer2, self.pool_tags2)

        self.flat_layers = self.pool_layer2.reshape((self.cfg.batch_size, -1))
        self.H = np.matmul(self.flat_layers, self.W)
        self.O = softmax(self.H)

    def backprop(self, pad_inputs, labels):
        one_hot_labels = np.eye(10)[labels]
        last_derivation = self.O - one_hot_labels
        delta_W = np.matmul(self.flat_layers.T, last_derivation) / self.cfg.batch_size
        self.W = self.optimization(self.cfg.lr, self.W, delta_W)
        delta_flatten = np.matmul(last_derivation, self.W.T)
        delta_pool2 = delta_flatten.reshape(self.cfg.batch_size, self.conv_layer2.shape[1],
                                           self.pool_layer2_height, self.pool_layer2_width)
        recover_delta_pool2 = self.recover_pool(delta_pool2, self.pool_arg2, self.pool_tags2)
        delta_relu2 = self.act_deriv(recover_delta_pool2)

        delta_F2 = self.get_conv_layers(self.pool_layer1, delta_relu2)
        delta_F2 = np.sum(delta_F2, 0) / delta_F2.shape[0]
        self.F2 = self.optimization(1, self.F2, delta_F2)

        delta_relu2 = delta_relu2.reshape(-1,delta_relu2.shape[2],delta_relu2.shape[3])
        pad_delta_relu2 = np.zeros((delta_relu2.shape[0],
                                   delta_relu2.shape[1]+2*(self.F2.shape[1]-1),
                                   delta_relu2.shape[2]+2*(self.F2.shape[2]-1)))
        for i in range(delta_relu2.shape[0]):
            pad_delta_relu2[i] = np.pad(delta_relu2[i],((self.F2.shape[1]-1,self.F2.shape[1]-1),
                                                        (self.F2.shape[2]-1,self.F2.shape[2]-1)),'constant')
        pad_delta_relu2 = pad_delta_relu2.reshape(self.cfg.batch_size, self.conv_arg2[0],
                                                  pad_delta_relu2.shape[1],pad_delta_relu2.shape[2])
        delta_pool1 = self.get_conv_layers(pad_delta_relu2,np.rot90(self.F2,2,(1,2)))
        delta_pool1 = np.sum(delta_pool1,1)/delta_pool1.shape[1]
        delta_pool1 = np.tile(np.expand_dims(delta_pool1,1),[1,self.conv_arg1[0],1,1])
        recover_delta_pool1 = self.recover_pool(delta_pool1, self.pool_arg1, self.pool_tags1)
        delta_relu1 = self.act_deriv(recover_delta_pool1)

        delta_F1 = self.get_conv_layers(pad_inputs, delta_relu1)
        delta_F1 = np.sum(delta_F1, 0) / delta_F1.shape[0]
        self.F1 = self.optimization(1, self.F1, delta_F1)

    def predict(self, inputs):
        self.forward(inputs)
        return np.argmax(self.O, 1)
