import numpy as np
import math

class relu:
    def __init__(self):
        pass

    def forward(self,u):
        self.u = u
        self.v = np.array([])
        for i in u:
            self.v = np.append(self.v, np.maximum(0,i))
        return self.v
    
    def backward(self, grad):
        self.w = np.array([])
        for i in grad:
            if i >= 0:
                self.w = np.append(self.w, 1)
            else:
                self.w = np.append(self.w, 0)
        return self.w
class sigmoid:
    def __init__(self):
        pass
    
    def forward(self, u):
        self.u = u
        self.v = np.array([])
        for i in self.u:
            if i >= 0:
                self.v = np.append(self.v, 1/(1 + math.exp(-i)))
            elif i < 0:
                self.v = np.append(self.v, math.exp(i)/(1+math.exp(i)))
        return self.v
    
    def backward(self, grad):
        self.w = 0

class tanh:
    def __init__(self):
        pass
    
    def forward(self, u):
        self.u = u
        self.v = np.array([])
        for i in self.u:
            if i >= 0:
                self.v = np.append(self.v, 2/(1 + math.exp(-2*i)) - 1)
            elif i < 0:
                self.v = np.append(self.v, 2*math.exp(2*i)/(1 + math.exp(2*i)) - 1)
        return self.v
    
class leaky_relu:
    def __init__(self):
        pass
    
    def forward(self, u):
        self.u = u
        self.v = np.array([])
        for i in self.u:
            if i >= 0:
                self.v = np.append(self.v, i)
            elif i < 0:
                self.v = np.append(self.v, 0.01*i)
            print(self.u)
            print(self.v)
        return self.v
    
class softmax:
    def __init__(self):
        pass
    
    def forward(self, u):    
        self.u = u
        self.v = []
        denominator = 0
        for i in self.u:
            denominator += np.sum(math.exp(i))
        for j in self.u:
            self.v.append(math.exp(j)/denominator)
        return self.v