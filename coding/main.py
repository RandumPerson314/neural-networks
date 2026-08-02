from neural_network import activations 
from neural_network import loss_functions 
from neural_network import layers
import numpy as np

relu = activations.relu
sigmoid = activations.sigmoid
tanh = activations.tanh
softmax = activations.softmax
u = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ])
v = np.array([
    [0],
    [1],
    [1],
    [0]])

print(f"start with {u}")
print(f"we want {v}")

input_layer = layers.InputLayer(u, 32, activations.relu())
input_layer_fp = input_layer.forward_pass()

hidden_layer_1 = layers.HiddenLayer(32, 10, activations.leaky_relu())
hidden_layer_1_fp = hidden_layer_1.forward_pass(input_layer_fp)

output_layer = layers.OutputLayer(10,v,activations.sigmoid())
output_layer_fp = output_layer.forward_pass(hidden_layer_1_fp) 
print(f"first try: {output_layer_fp}")

for i in range(8888):
    output_layer_bp = output_layer.backward_pass(0.5)
    hidden_layer_1_bp = hidden_layer_1.backward_pass(output_layer, 0.2)
    input_layer_bp = input_layer.backward_pass(hidden_layer_1, 0.1)
    input_layer_fp = input_layer.forward_pass()
    hidden_layer_1_fp = hidden_layer_1.forward_pass(input_layer_fp)
    output_layer_fp = output_layer.forward_pass(hidden_layer_1_fp)
    if i % 500 == 0:
        print(i, loss_functions.MSE(output_layer_fp, v))

print(f"last try {output_layer_fp}")