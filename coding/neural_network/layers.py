import numpy as np
from neural_network import parameters
from neural_network import activations
rng = np.random.default_rng()
activations_dict = {
    "ReLU": activations.relu,
    "Softmax": activations.softmax,
    "Sigmoid": activations.sigmoid,
    "Tanh": activations.tanh,
    "Leaky ReLU": activations.leaky_relu
    }

class output_layer:
    def __init__ (self, input_size, n_neurons, activation):
        self.input_size = input_size
        self.n_neurons = n_neurons

        self.weights = parameters.weight_matrix(n_neurons, input_size, "random")
        self.bias = parameters.bias(n_neurons, "random")
        self.activation = activation

        
    
    def forward_pass(self, input):
        self.output = self.weights @ input + self.bias
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, target, learning_rate):
        self.error = target - self.activated_output
        self.delta = self.error * self.activation.backward(self.activated_output)
        self.weights += (self.activated_output.T @ self.delta) * learning_rate
        self.bias += (np.sum(self.delta, axis=0, keepdims=True) * learning_rate)
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