import numpy as np

class SGD:
    def __init__(self):
        pass

    def optimize(self, theta, lr, derivation):
        return theta - lr*derivation

class Momentum:
    def __init__(self):
        self.v = None
        self.alpha = 0.9

    def optimize(self, theta, lr, derivation):
        if self.v is None:
            self.v = -lr * derivation
        else:
            self.v = self.alpha * self.v - lr * derivation
        return theta + self.v

class AdaGrad:
    def __init__(self):
        self.r = None
        self.gama = 1e-7

    def optimize(self, theta, lr, derivation):
        if self.r is None:
            self.r = derivation * derivation
        else:
            self.r = self.r + derivation* derivation
        delta_theta = -lr * (1 /(self.gama + np.sqrt(self.r))) * derivation
        return theta + delta_theta

class RMSProp:
    def __init__(self):
        self.r = None
        self.gama = 1e-6
        self.rou = 0.99

    def optimize(self, theta, lr, derivation):
        if self.r is None:
            self.r = (1-self.rou)*derivation*derivation
        else:
            self.r = self.rou*self.r + (1-self.rou)*derivation*derivation
        delta_theta = -lr*(1/np.sqrt(self.gama+self.r))*derivation
        return theta + delta_theta

class Adam:
    def __init__(self):
        self.s = None
        self.r = None
        self.rou1 = 0.9
        self.rou2 = 0.999
        self.gama = 1e-8

    def optimize(self, theta, lr, derivation, t=None):
        if self.s is None:
            self.s  = (1-self.rou1)*derivation
        else:
            self.s = self.rou1 * self.s + (1-self.rou1)*derivation
        if self.r is None:
            self.r = (1-self.rou2)*derivation*derivation
        else:
            self.r = self.rou2*self.r + (1-self.rou2)*derivation*derivation
        modified_s = self.s/(1-np.power(self.rou1, t))
        modified_r = self.r/(1-np.power(self.rou2, t))
        delta_theta = -lr*modified_s/(np.sqrt(modified_r)+derivation)
        return theta + delta_theta


