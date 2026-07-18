from neural_network import activations 
from neural_network import loss_functions 
from neural_network import layers
import numpy as np

relu = activations.relu
sigmoid = activations.sigmoid
tanh = activations.tanh
softmax = activations.softmax
u = np.random.rand(10, 2)
v = np.random.rand(2, 2)

print(f"start with {u}")
print(f"we want {v}")

output_layer = layers.output_layer(np.shape(u)[0],np.shape(v)[0],activations.sigmoid())
output_layer_fp = output_layer.forward_pass(u)
print(f"first try: {output_layer_fp}")

for i in range(888888):
    output_layer_bp = output_layer.backward_pass(v, 0.01)
    output_layer_fp = output_layer.forward_pass(u)

print(f"last try {output_layer_fp}")