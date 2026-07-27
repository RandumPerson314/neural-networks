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

input_layer = layers.input_layer(u, 10, activations.sigmoid())
input_layer_fp = input_layer.forward_pass()

output_layer = layers.output_layer(10,v,activations.sigmoid())
output_layer_fp = output_layer.forward_pass(input_layer_fp) 
print(f"first try: {output_layer_fp}")

for i in range(8888):
    output_layer_bp = output_layer.backward_pass(0.01)
    input_layer_bp = input_layer.backward_pass(output_layer, 0.01)
    input_layer_fp = input_layer.forward_pass()
    output_layer_fp = output_layer.forward_pass(input_layer_fp)

print(f"last try {output_layer_fp}")