import numpy as np
from neural_network import parameters
from neural_network import activations

class InputLayer:
    def __init__ (self, input_size, target_size, activation):
        self.input_size = input_size
        self.target_size = target_size
        self.weights = parameters.weight_matrix(self.input_size + 1, self.target_size, "xavier_uniform")
        self.activation = activation
    
    def forward_pass(self, input):
        self.input = np.hstack([input, np.ones((np.shape(input)[0],1))])
        self.output = np.dot(self.input, self.weights)
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, next_layer, learning_rate):
        self.error = np.dot(next_layer.delta, next_layer.old_weights[:-1].T)
        self.delta = self.error * self.activation.backward(self.activated_output)
        self.grad = np.dot(self.input.T, self.delta) / self.input.shape[0]
        self.weights -= self.grad * learning_rate
        return(self.delta)

class HiddenLayer:
    def __init__ (self, input_size, target_size, activation):
        self.input_size = input_size
        self.target_size = target_size
        self.weights = parameters.weight_matrix(self.input_size + 1, self.target_size, "he_normal")
        self.activation = activation
    
    def forward_pass(self, input):
        self.input = np.hstack([input, np.ones((np.shape(input)[0],1))])
        self.output = np.dot(self.input, self.weights)
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, next_layer, learning_rate):
        self.error = np.dot(next_layer.delta, next_layer.old_weights[:-1].T)
        self.delta = self.error * self.activation.backward(self.activated_output)
        self.old_weights = self.weights.copy()
        self.grad = np.dot(self.input.T, self.delta) / self.input.shape[0]
        self.weights -= self.grad * learning_rate
        return(self.delta)

    
class OutputLayer:
    def __init__ (self, input_size, target_size, activation):
        self.input_size = input_size
        self.target_size = target_size
        self.weights = parameters.weight_matrix(self.input_size + 1, self.target_size, "xavier_uniform")
        self.activation = activation
    
    def forward_pass(self, input):
        self.input = np.hstack([input, np.ones((np.shape(input)[0],1))])
        self.output = np.dot(self.input, self.weights)
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, target, learning_rate):
        self.target = target
        self.error =  self.activated_output - self.target
        self.delta = self.error * self.activation.backward(self.activated_output)
        self.old_weights = self.weights.copy()
        self.grad = np.dot(self.input.T, self.delta) / self.input.shape[0]
        self.weights -= self.grad * learning_rate
        return(self.delta)