from neural_network import activations 
from neural_network import loss_functions 
from neural_network import layers
import numpy as np

relu = activations.relu
sigmoid = activations.sigmoid
tanh = activations.tanh
softmax = activations.softmax
u = np.zeros(32)
v = np.zeros(10)

for i in range(10):
    v[i] = np.random.default_rng().random()

layer_1 = layers.hidden_layer(10, 32, activations.relu())
layer_1_forward_pass = layer_1.forward_pass(v)

layer_1_error = loss_functions.MSE(layer_1_forward_pass, u)


print(layer_1_forward_pass)
print(layer_1_error)