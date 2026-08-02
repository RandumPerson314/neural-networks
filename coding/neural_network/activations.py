import numpy as np
import math

class relu:
    def __init__(self):
        pass

    def forward(self,u):
        self.v = np.maximum(0,u)
        return self.v
    
    def backward(self, grad):
        self.w = self.v > 0
        return self.w

class sigmoid:
    def __init__(self):
        pass
    
    def forward(self, u):
        self.v = 1/(1 + np.exp(-u))
        return self.v
    
    def backward(self, activated_output):
        self.w = (activated_output * (1 - activated_output))
        return(self.w)

class tanh:
    def __init__(self):
        pass
    
    def forward(self, u):
        self.v = np.tanh(u)
        return self.v

    def backward(self, activated_output):
        self.w = 1 - self.v**2
        return self.w
    
class leaky_relu:
    def __init__(self):
        pass
    
    def forward(self, u):
        self.u = u
        self.v = np.where(self.u > 0, self.u, 0.01 * self.u)
        return self.v

    def backward(self, activated_output):
        self.w = np.where(self.u > 0, 1 , 0.01)
        return self.w
    
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

class linear:
    def forward(self, u):
        self.v = u
        return self.v

    def backward(self, activated_output):
        return 1.0