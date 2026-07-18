import numpy as np
from neural_network import parameters
from neural_network import activations

def write_to_debug(text):
    with open("debug.txt", "a") as f:
        f.write(text + "\n")

class output_layer:
    def __init__ (self, input_size, n_neurons, activation):
        self.input_size = input_size
        self.n_neurons = n_neurons

        self.weights = parameters.weight_matrix(n_neurons, input_size + 1, "xavier_uniform")
        self.bias = parameters.bias(n_neurons, "random")
        self.activation = activation
    
    def forward_pass(self, input):
        self.input = input
        self.batch_size = input.shape[1]
        self.input = np.vstack([input, np.ones(self.batch_size)])
        self.output = self.weights @ self.input 
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, target, learning_rate):
        self.error = target - self.activated_output
        self.delta = self.error * self.activation.backward(self.output)
        self.weights += np.dot(self.delta, self.input.T) * learning_rate
        return(self.delta)

class hidden_layer:
    def __init__ (self, input_size, n_neurons, activation):
        self.input_size = input_size
        self.n_neurons = n_neurons

        self.weights = parameters.weight_matrix(n_neurons, input_size, "random")
        self.bias = parameters.bias(n_neurons, "random")
        self.activation = activation
    
    def forward_pass(self, input):
        self.output = self.weights @ input + self.bias

        return(self.activation.forward(self.output))
    
    def backward_pass(self, target):
        self.error = target - self.activated_output
        self.delta = self.error * self.activation.backward(self.activated_output)
        return(self.delta)