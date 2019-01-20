# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W = tf.Variable(tf.random_normal(shape=[7*7*16, 10], mean=0, stddev=0.1), name='W')
        self.global_step = tf.get_variable(
            'global_step', shape=[], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)

        self.img = tf.placeholder(tf.float32, [None,28,28], name='img')
        self.img = tf.expand_dims(self.img,axis = -1)
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.update_tensor_and_opt()

    def build_net(self):
        self.conv1 = tf.layers.conv2d(self.img, 16, 3, padding='same', activation=tf.nn.elu)
        self.pool1 = tf.layers.max_pooling2d(self.conv1, 2, 2)
        self.conv2 = tf.layers.max_pooling2d(self.pool1, 32, 3, 3, padding='same', activation=tf.nn.elu)
        self.pool2 = tf.layers.max_pooling2d(self.conv2, 2, 2)
        self.flat = tf.reshape(self.pool2, [-1, 7*7*16])
        self.output = tf.matmul(self.flat, self.W)

    def build_loss(self):

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.label, logits=self.output)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.output, -1), tf.int32),
            self.label,
        )
        return tf.cast(correct, tf.float32)

    def update_tensor_and_opt(self):
        self.build_net()
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()
        self.opt = tf.train.AdamOptimizer(self.cfg.lr)
        self.train_op = self.opt.minimize(self.loss, self.global_step,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def get_feed_dict(self, x_train, y_train):
        x_train = np.expand_dims(x_train,-1)
        feed_dict = {
            self.img : x_train,
            self.label : y_train,
        }
        return feed_dict

    def train_step(self,sess, x_train,y_train):
        feed_dict = self.get_feed_dict(x_train,y_train)
        loss, train_op, accuracy = sess.run([self.loss, self.train_op, self.accuracy],
                                            feed_dict = feed_dict)
        return loss, accuracy

    def update_learing_rate(self,i):
        if i <= 30:
            self.cfg.lr = (1 - i / 30) * 0.1 + (i / 30) * 0.01 * 0.1


