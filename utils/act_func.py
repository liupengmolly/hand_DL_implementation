import numpy as np

def sigmoid(h):
    return 1/(1+np.exp(-h))

def sigmoid_deriv(o):
    return o*(1-o)

def relu(h):
    return np.where(h>0,h,np.zeros(h.shape))

def relu_deriv(o):
    return np.where(o>0,np.ones(o.shape),np.zeros(o.shape))

def elu(h):
    return np.where(h>0,h,np.exp(h)-1)

def elu_deriv(o):
    return np.where(o>0,np.ones(o.shape),np.exp(o))

def leakyrelu(h):
    return np.where(h>0,h,0.01*h)

def leakyrelu_deriv(o):
    return np.where(o>0,np.ones(o.shape),0.01)

def tanh(h):
    return (np.exp(h)-np.exp(-h))/(np.exp(h)+np.exp(-h))

def tanh_deriv(o):
    return 1-o*o

def softmax(h):
    h = np.exp(h-np.expand_dims(np.max(h,1),1))
    h_sum = np.sum(h,1)
    h = h/np.expand_dims(h_sum,1)
    return h

def cross_entropy_loss(o,labels,batch_size):
    one_hot_labels = np.eye(10)[labels]
    loss = np.sum(-np.sum(one_hot_labels*(np.log(o)),1))/batch_size
    return loss

def linear_decay_lr(k,lr,init_lr):
    if k<=100:
        lr = (1-k/100)*init_lr+(k/100)*0.01*init_lr
    return lr
