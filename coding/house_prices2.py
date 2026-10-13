import neural_network.activations as activations
import neural_network.layers as layers
import neural_network.loss_functions as loss_functions
import neural_network.parameters as parameters
import numpy as np

epochs = 1000
batch_size = 64
neurons_1 = 32
neurons_2 = 32
neurons_3 = 32

X_train = np.load("data/ames_X_train.npy")
X_test = np.load("data/ames_X_test.npy")
y_train = np.load("data/ames_y_train.npy")
y_test = np.load("data/ames_y_test.npy")

features = np.shape(X_train)[1]

# Train model
input_layer = layers.InputLayer(features, neurons_1, activations.relu)
hidden_layer_1 = layers.HiddenLayer(neurons_1, neurons_2, activations.relu)
hidden_layer_2 = layers.HiddenLayer(neurons_2, neurons_3, activations.sigmoid)
output_layer = layers.OutputLayer(neurons_3, batch_size, activations.relu)

for epoch in range(epochs):
    forward = output_layer.forward_pass(
        hidden_layer_2.forward_pass(
            hidden_layer_1.forward_pass(
                input_layer.forward_pass(X_train)
            )
        )
    )
    
    input_layer.backward_pass(
        hidden_layer_1.backward_pass(
            hidden_layer_2.backward_pass(
                output_layer.backward_pass(forward)
            )
        )
    )
    