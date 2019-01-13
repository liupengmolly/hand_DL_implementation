# -*- coding:utf-8 -*-
import sys
import os
import random
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess.io import *
from preprocess.config import cfg
from model.NN import NN
from model.CNN import CNN
from preprocess.io import load_param,dump_param
from model.util import *

prefix =''
if cfg.env == 'pycharm':
    prefix = './'

logging.basicConfig(filename = prefix+'log/{}_{}_{}_{}_{}_{}.log'.format(cfg.model_name,cfg.act_func,
                                                                      cfg.batch_size,cfg.layers_num,
                                                                      cfg.units_num, cfg.lr),
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
        if 'cnn' in cfg.model_name:
            yield data[next_index:]+\
                [(np.zeros((30, 30)),0) for _ in range(cfg.batch_size-len(data)+next_index)]
        else:
            yield data[next_index:]+[(np.zeros(784),0) for _ in range(cfg.batch_size-len(data)+next_index)]

def cal_ac(nn,test):
    predicts = []
    x_test,y_test = zip(*test)
    x_test,y_test = np.array(x_test),np.array(y_test)
    for batch in get_batches(test,shuffle=False):
        x_batch,y_batch = zip(*batch)
        x_batch,y_batch = np.array(x_batch),np.array(y_batch)
        predicts.append(nn.predict(x_batch))
    #get_batches得到的最后一个batch因为补全可能会重复用到前面的数据
    predicts = np.concatenate(tuple(predicts))[:len(test)]
    ac = np.sum(predicts==y_test)/len(test)
    return ac

if __name__ == '__main__':
    images, labels = load_mnist(prefix)
    x_test,y_test = load_mnist(prefix,'test')
    # if cfg.act_func != 'sigmoid':
    images = images/np.expand_dims(np.sum(images,1),1)
    x_test = x_test/np.expand_dims(np.sum(x_test,1),1)
    if 'cnn' in cfg.model_name:
        images = np.reshape(images,(60000,28,28))
        x_test = np.reshape(x_test,(10000,28,28))

        pad_images = np.zeros((60000,30,30))
        pad_test = np.zeros((10000,30,30))
        for i in range(len(images)):
            pad_images[i] = np.pad(images[i],((1,1),(1,1)),'constant')
        for i in range(len(x_test)):
            pad_test[i] = np.pad(x_test[i], ((1,1),(1,1)), 'constant')
        images, x_test = pad_images, pad_test
        model = CNN(cfg,[cfg.batch_size,30,30],[4,3,3],[2,2,2],[8,3,3],[4,4,4])
    else:
        model = NN(cfg)

    train_data = list(zip(list(images),list(labels)))
    test_data = list(zip(list(x_test),list(y_test)))

    model_file = prefix+'data/models/{}_{}_{}_{}_{}'.format(cfg.model_name,cfg.act_func,cfg.batch_size,
                                                     cfg.layers_num,cfg.units_num)
    best_model = model
    ac = 0
    i = 0
    early_stop = 0
    for batch in get_batches(train_data,10000):
        x_train,y_train = zip(*batch)
        x_train,y_train = np.array(x_train),np.array(y_train)
        model.forward(x_train)
        model.backprop(x_train,y_train)
        i = i+1
        if i%937 == 0:
            cur_ac = cal_ac(model,test_data)
            logging.info('{}: {}'.format(i/937,cur_ac))
            if  cur_ac > ac:
                ac = cur_ac
                early_stop = 0
                best_model = model
            else:
                early_stop += 1
            if early_stop == 30:
                logging.info('early stop,highest accuracy is {}'.format(ac))
                dump_param(best_model,model_file)
                break
            model.cfg.lr = line_decay_lr(i%937,model.cfg.lr,model.init_lr)



