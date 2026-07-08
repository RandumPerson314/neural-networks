import activations
import layers
import numpy as np
relu = activations.relu
sigmoid = activations.sigmoid
tanh = activations.tanh
softmax = activations.softmax
test_weights = np.array([[1,2,2],[4,3,2],[2,4,5]])
test_input = np.array([1,6,3])
test_bias = 1
tester = layers.output_layer(test_input, test_weights, test_bias, relu)

print(tester.output())