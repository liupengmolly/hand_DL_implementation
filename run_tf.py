# -*- coding:utf-8 -*-sigmoid
import sys
import os
import random
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.io import *
from utils.config import cfg
from tf.CNN import CNN
from utils.act_func import *
import tensorflow as tf

prefix =''
if cfg.env == 'pycharm':
    prefix = './'

logging.basicConfig(filename = prefix+'tflog/{}_ch16_k5_{}_{}_{}_{}.log'
                    .format(cfg.model_name,cfg.act_func,cfg.batch_size,cfg.units_num, cfg.lr),
                    filemode= 'w',
                    format = '%(asctime)s-%(name)s-%(levelname)s-%(message)s',
                    level = logging.INFO)

def get_batches(data,round = 1,shuffle = True):
    for _ in range(round):
        if shuffle:
            random.shuffle(data)
        cur_index,next_index = None,None
        for batch_index in range(int(len(data)/cfg.batch_size)):
            cur_index = cfg.batch_size * batch_index
            next_index = cfg.batch_size * (batch_index+1)
            yield data[cur_index:next_index]
        yield data[next_index:]

def cal_ac(sess, model, test):
    acc_array = []
    for batch in get_batches(test,shuffle=False):
        x_batch,y_batch = zip(*batch)
        x_batch,y_batch = np.array(x_batch),np.array(y_batch)
        loss, acc = model.train_step(sess,x_batch,y_batch)
        acc_array.extend(acc)
    ac = sum(acc_array)/len(acc_array)
    return ac

if __name__ == '__main__':
    images, labels = load_mnist(prefix)
    x_test,y_test = load_mnist(prefix,'test')
    images = np.reshape(images,(60000,28,28))
    x_test = np.reshape(x_test,(10000,28,28))

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions()
        graph_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=graph_config)
        with sess.as_default():
            with tf.variable_scope("test") as scope:
                model = CNN(cfg)
        sess.run(tf.global_variables_initializer())

        train_data = list(zip(list(images),list(labels)))
        test_data = list(zip(list(x_test),list(y_test)))

        ac = 0
        loss = 0
        i = 0
        early_stop = 0
        for batch in get_batches(train_data,10000):
            x_train,y_train = zip(*batch)
            x_train,y_train = np.array(x_train),np.array(y_train)
            train_loss, train_acc = model.train_step(sess,x_train,y_train)
            i = i+1
            if i%937 == 0:
                cur_ac = cal_ac(sess, model,test_data)
                logging.info('{}: {}'.format(i/937,cur_ac))
                if  cur_ac > ac:
                    ac = cur_ac
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop == 30:
                    logging.info('early stop,highest accuracy is {}'.format(ac))
                    break
                model.update_learing_rate(int(i/937))

