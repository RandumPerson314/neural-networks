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

for i in range(32):
    u[i] = np.random.default_rng().random()

for i in range(10):
    v[i] = np.random.default_rng().random()

print(f"start with {u}")
print(f"we want {v}")

output_layer = layers.output_layer(len(u),len(v),activations.sigmoid())
output_layer_fp = output_layer.forward_pass(u)
print(f"first try: {output_layer_fp}")

for i in range(1000):
    output_layer_bp = output_layer.backward_pass(v, 0.01)
    output_layer_fp = output_layer.forward_pass(u)

print(f"last try {output_layer_fp}")