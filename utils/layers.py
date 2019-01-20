import numpy as np
import math

class Conv_2d:
    def __init__(self, inputs, conv_arg):
        """
        卷积层, 卷积移动的方式暂时只支持‘same’模式

        :param inputs_shape: 输入数据的相关数据，(batch_size,height,width)
        :param conv_arg:  卷积层的相关参数, (channels_num, height, width)
        """
        if conv_arg[1] != conv_arg[2]:
            yield ValueError("the filter height must be same with the width, while the heights is"
                              " {} and the width is {}".format(conv_arg[1], conv_arg[2]))
        self.half_F = int((conv_arg[1]-1)/2)
        self.inputs = inputs
        self.conv_arg = conv_arg
        self.F = np.random.uniform(-1, 1, conv_arg)
        self.layers = np.zeros((inputs.shape[0], conv_arg[0],
                                inputs.shape[-2]-2*self.half_F, inputs.shape[-1]-2*self.half_F))

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
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs,1)

        for j in range(self.half_F, inputs.shape[-2] - self.half_F):
            for k in range(self.half_F, inputs.shape[-1] - self.half_F):
                self.layers[:, :, j-self.half_F, k-self.half_F] = self.conv_cal(inputs, j, k, F)

    def forward(self):
        self.get_conv_layers(self.inputs, self.F)