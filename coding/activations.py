import numpy as np
import math

def relu (u):
    v = np.array([])
    for i in u:
        v = np.append(v, np.maximum(0,i))
    return v

def sigmoid(u):
    v = np.array([])
    for i in u:
        if i >= 0:
            v = np.append(v, 1/(1 + math.exp(-i)))
        elif i < 0:
            v = np.append(v, math.exp(i)/(1+math.exp(i)))
    return v
    
def tanh(u):
    v = np.array([])
    for i in u:
        if i >= 0:
            v = np.append(v, 2/(1 + math.exp(-2*i)) - 1)
        elif i < 0:
            v = np.append(v, 2*math.exp(2*i)/(1 + math.exp(2*i)) - 1)
    return v

def leaky_relu(u):
    if u >= 0:
        return(u)
    elif u < 0:
        return(0.01*u)
    
def softmax (u):
    v = []
    denominator = 0
    for i in u:
        denominator += np.sum(math.exp(i))
    for j in u:
        v.append(math.exp(j)/denominator)
    return v