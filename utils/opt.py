import numpy as np

class SGD:
    def __init__(self):
        pass

    def optimize(self, theta, lr, derivation,i=None):
        return theta - lr*derivation

class Momentum:
    def __init__(self):
        self.v = []
        self.alpha = 0.9

    def optimize(self, theta, lr, derivation,i):
        if i>=len(self.v):
            self.v.extend([None for ii in range(i+1-len(self.v))])
        if self.v[i] is None:
            self.v[i] = -lr * derivation
        else:
            self.v[i] = self.alpha * self.v[i] - lr * derivation
        return theta + self.v[i]

class AdaGrad:
    def __init__(self):
        self.rs = []
        self.gama = 1e-7

    def optimize(self, theta, lr, derivation,i):
        if i>=len(self.rs):
            self.rs.extend([None for ii in range(i+1-len(self.rs))])
        if self.rs[i] is None:
            self.rs[i] = derivation * derivation
        else:
            self.rs[i] = self.rs[i] + derivation* derivation
        delta_theta = -lr * (1 /(self.gama + np.sqrt(self.rs[i]))) * derivation
        return theta + delta_theta

class RMSProp:
    """
    RMSProp有问题
    """
    def __init__(self):
        self.rs = []
        self.gama = 1e-7
        self.rou = 0.9

    def optimize(self, theta, lr, derivation,i):
        if i>=len(self.rs):
            self.rs.extend([None for ii in range(i+1-len(self.rs))])
        if self.rs[i] is None:
            self.rs[i] = (1-self.rou)*derivation*derivation
        else:
            self.rs[i] = self.rou*self.rs[i] + (1-self.rou)*derivation*derivation
        delta_theta = -lr*(1/(np.sqrt(self.rs[i])+self.gama))*derivation
        return theta + delta_theta

class Adam:
    def __init__(self):
        self.ss = []
        self.rs = []
        self.rou1 = 0.9
        self.rou2 = 0.999
        self.gama = 1e-8
        self.t = 1

    def optimize(self, theta, lr, derivation,i):
        if i>=len(self.ss):
            self.ss.extend([None for ii in range(i+1-len(self.ss))])
        if self.ss[i] is None:
            self.ss[i] = (1-self.rou1)*derivation
        else:
            self.ss[i] = self.rou1 * self.ss[i] + (1-self.rou1)*derivation
        if i>=len(self.rs):
            self.rs.extend([None for ii in range(i+1-len(self.rs))])
        if self.rs[i] is None:
            self.rs[i] = (1-self.rou2)*derivation*derivation
        else:
            self.rs[i] = self.rou2*self.rs[i] + (1-self.rou2)*derivation*derivation
        modified_s = self.ss[i]/(1-np.power(self.rou1, self.t))
        modified_r = self.rs[i]/(1-np.power(self.rou2, self.t))
        #最后一个更新的参数即为结束一轮更新
        if i==0:
            self.t += 1
        delta_theta = -lr*modified_s/(np.sqrt(modified_r)+1e-8)
        return theta + delta_theta


