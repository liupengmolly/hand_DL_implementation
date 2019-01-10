import numpy as np

def sigmoid(h):
    return 1/(1+np.exp(-h))

def sigmoid_deriv(o):
    return o*(1-o)

def relu(h):
    return np.where(h>0,h,np.zeros(h.shape))
    # return np.where(h<10,h,np.ones(h.shape)*10)

def relu_deriv(o):
    return np.where(o>0,np.ones(o.shape),np.zeros(o.shape))

def sgd(lr,theta,derivation):
    return theta - lr*derivation


