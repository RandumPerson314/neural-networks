import numpy as np
class output_layer:
    def __init__ (self, input, weights, bias, activation):
        self.input = input
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def update(self, new_weights, new_bias):
        self.weights = new_weights
        self.bias = new_bias
    
    def output(self):
        out = self.weights @ self.input
        return(self.activation(out))

class hidden_layer:
    def __init__ (self, input, weights, bias, activation):
        self.input = input
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def update(self, new_weights, new_bias):
        self.weights = new_weights
        self.bias = new_bias
    
    def output(self):
        out = self.weights @ self.input
        return(self.activation(out))