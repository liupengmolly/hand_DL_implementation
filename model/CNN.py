from utils.act_func import *
from utils.layers import *
from utils.opt import *

class CNN:
    def __init__(self, cfg, inputs_shape, conv_arg1, pool_arg1, conv_arg2=None, pool_arg2=None):
        self.cfg = cfg
        self.inputs_shape = inputs_shape
        self.act_func = eval(self.cfg.act_func)
        self.act_deriv = eval(self.cfg.act_func+'_deriv')
        self.optimization = eval(cfg.optimization+'()')
        self.init_lr = cfg.lr

        self.conv1 = Conv_2d(cfg, inputs_shape, conv_arg1)
        self.pool1 = MaxPool(cfg, self.conv1.layers.shape, pool_arg1)

        # self.conv2 = Conv_2d(cfg, self.pool1.pool_layer.shape, conv_arg2)
        # self.pool2 = MaxPool(cfg, self.conv2.layers.shape, pool_arg2)

        flat_length = conv_arg1[-1] * self.pool1.pool_layer_height * self.pool1.pool_layer_width
        self.W = np.random.uniform(-1,1,(flat_length,10))
        self.H = np.zeros((self.cfg.batch_size, 10))
        self.O = np.zeros((self.cfg.batch_size, 10))

    def forward(self, pad_inputs):
        conv_layers1 = self.conv1.forward(pad_inputs)
        relu_layers1 = self.act_func(conv_layers1)
        pool_layer1, self.pool1_tags = self.pool1.forward(relu_layers1)

        # conv_layers2 = self.conv2.forward(pool_layer1)
        # relu_layers2 = self.act_func(conv_layers2)
        # pool_layer2, self.pool2_tags = self.pool2.forward(relu_layers2)

        self.flat_layers = pool_layer1.reshape((self.cfg.batch_size, -1))
        self.H = np.matmul(self.flat_layers, self.W)
        self.O = softmax(self.H)

    def backprop(self, pad_inputs, labels):
        one_hot_labels = np.eye(10)[labels]
        last_derivation = self.O - one_hot_labels
        delta_W = np.matmul(self.flat_layers.T, last_derivation) / self.cfg.batch_size
        self.W = self.optimization.optimize(self.W, self.cfg.lr, delta_W)
        delta_flatten = np.matmul(last_derivation, self.W.T)
        delta_pool1 = delta_flatten.reshape(self.cfg.batch_size,
                                            self.pool1.pool_layer_height,
                                            self.pool1.pool_layer_width,
                                            self.pool1.inputs_shape[-1])
        # delta_pool2 = self.pool2.backprop(delta_pool2, self.pool2_tags)
        # delta_relu2 = self.act_deriv(delta_pool2)
        # delta_conv2 = self.conv2.backprop(self.pool1.pool_layer, self.cfg.lr, delta_relu2, True)

        delta_pool1 = self.pool1.backprop(delta_pool1, self.pool1_tags)
        delta_relu1 = self.act_deriv(delta_pool1)
        _ = self.conv1.backprop(pad_inputs, self.cfg.lr, delta_relu1)

    def predict(self, inputs):
        self.forward(inputs)
        return np.argmax(self.O, 1)