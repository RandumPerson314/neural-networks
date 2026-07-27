import numpy as np
from neural_network import parameters
from neural_network import activations

def write_to_debug(text):
    with open("debug.txt", "a") as f:
        f.write(text + "\n")

class output_layer:
    def __init__ (self, input_size, target, activation):
        self.input_size = input_size
        self.target = target
        self.target_size = np.shape(self.target)[1]

        self.weights = parameters.weight_matrix(self.input_size + 1, self.target_size, "xavier_uniform")
        self.activation = activation
        print(f"weights: {self.weights}")
    
    def forward_pass(self, input):
        self.input = np.hstack([input, np.ones((np.shape(input)[0],1))])
        # print(self.input)
        self.output = np.dot(self.input, self.weights)
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, learning_rate):
        self.error = self.target - self.activated_output
        self.delta = self.error * self.activation.backward(self.output)
        self.weights += np.dot(self.input.T, self.delta) * learning_rate
        return(self.delta)

class input_layer:
    def __init__ (self, input, target_size, activation):
        self.input = input
        self.input_size = np.shape(self.input)[1]
        self.target_size = target_size

        self.weights = parameters.weight_matrix(self.input_size + 1, self.target_size, "xavier_uniform")
        self.activation = activation
        print(f"weights: {self.weights}")
    
    def forward_pass(self):
        self.input = np.hstack([self.input, np.ones((np.shape(self.input)[0],1))])
        # print(self.input)
        self.output = np.dot(self.input, self.weights)
        self.activated_output = self.activation.forward(self.output)
        return(self.activated_output)
    
    def backward_pass(self, next_layer, learning_rate):
        print(f"next layer delta: {next_layer.delta.shape}")
        print(f"next layer weights T: {next_layer.weights.T.shape}")
        self.error = np.dot(next_layer.delta, next_layer.weights[:-1].T)
        print(f"self error shape: {self.error.shape}")
        print(f"backward activation: {self.activation.backward(self.output).shape}")
        self.delta = self.error * self.activation.backward(self.output)
        print(f"input shape: {self.input.T.shape}")
        self.weights += np.dot(self.input.T, self.delta) * learning_rate
        return(self.delta)