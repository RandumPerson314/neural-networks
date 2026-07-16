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
        return(self.output)
    
    def backward_pass(self, x):
        return()

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
    
    def backward_pass(self):
        out = np.zeros(self.n_neurons)
        out = activations.sigmoid_prime()
        return(out)