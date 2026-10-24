import neural_network.activations as activations
import neural_network.layers as layers
import neural_network.loss_functions as loss_functions
import neural_network.parameters as parameters
import numpy as np
import matplotlib.pyplot as plt

epochs = 10000

batch_size = 64
neurons_1 = 32
neurons_2 = 32
neurons_3 = 32
lr = 0.2


X_train = np.load("data/ames_X_train.npy")
X_test = np.load("data/ames_X_test.npy")
y_train = np.load("data/ames_y_train.npy")
y_test = np.load("data/ames_y_test.npy")

# Standardize values
X_mean = np.mean(X_train, axis=0)
y_mean = np.mean(y_train, axis=0)
X_std = np.std(X_train, axis=0)
y_std = np.std(y_train, axis=0)

X_train = (X_train - X_mean)/X_std
X_test = (X_test - X_mean)/X_std
y_train = (y_train - y_mean)/y_std
y_test = (y_test - y_mean)/y_std

features = np.shape(X_train)[1]
output_dimension = np.shape(y_train)[1]

# Train model
input_layer = layers.InputLayer(features, neurons_1, activations.leaky_relu())
hidden_layer_1 = layers.HiddenLayer(neurons_1, neurons_2, activations.leaky_relu())
hidden_layer_2 = layers.HiddenLayer(neurons_2, neurons_3, activations.leaky_relu())
output_layer = layers.OutputLayer(neurons_1, output_dimension, activations.leaky_relu())

epochs_for_plot = []
errors_for_plot = []

for epoch in range(epochs):
    rng = np.random.default_rng()
    idx = rng.permutation(len(X_train))
    shuffled_X_train = X_train[idx]
    shuffled_y_train = y_train[idx]
    split_X_train = np.array_split(shuffled_X_train, range(batch_size, len(shuffled_X_train), batch_size))
    split_y_train = np.array_split(shuffled_y_train, range(batch_size, len(shuffled_y_train), batch_size))
    for batch_number in range(10, len(split_X_train)):
        forward = output_layer.forward_pass(
            hidden_layer_2.forward_pass(
                hidden_layer_1.forward_pass(
                    input_layer.forward_pass(split_X_train[batch_number])
                )
            )
        )
        output_layer.backward_pass(split_y_train[batch_number], lr)
        hidden_layer_2.backward_pass(output_layer, lr)
        hidden_layer_1.backward_pass(hidden_layer_2, lr)
        input_layer.backward_pass(output_layer, lr)
    if epoch % 200 == 0 and epoch != 0:
        guesses = output_layer.forward_pass(
            hidden_layer_2.forward_pass(
                hidden_layer_1.forward_pass(
                    input_layer.forward_pass(X_train)
                )
            )
        )
        epochs_for_plot.append(epoch)
        errors_for_plot.append(loss_functions.MSE(guesses, y_train))

train_guess = output_layer.forward_pass(
            hidden_layer_2.forward_pass(
                hidden_layer_1.forward_pass(
                    input_layer.forward_pass(X_train)
                )
            )
        )

test_guess = output_layer.forward_pass(
            hidden_layer_2.forward_pass(
                hidden_layer_1.forward_pass(
                    input_layer.forward_pass(X_test)
                )
            )
        )

# Unstandardize
train_guess = train_guess * y_std + y_mean
test_guess = test_guess * y_std + y_mean
y_train = y_train * y_std + y_mean
y_test = y_test * y_std + y_mean

print(f"Model's first five guesses on training data:\n {train_guess[:5].round()}")
print(f"Real first five house prices in training data:\n {y_train[:5]}")
print(f"Model's first five guesses on testing data:\n {test_guess[:5].round()}")
print(f"Real first five house prices in testing data:\n {y_test[:5]}")


x = np.arange(10)  # The label locations: [0, 1, 2, 3]
width = 0.35                    # The width of each individual bar

# 3. Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# =========================================================================
# CHART 1: Left Panel (ax1)
# =========================================================================
rects1_a = ax1.bar(x - width/2, train_guess.round().flatten()[:10], width, label='Model', color='#1f77b4')
rects1_b = ax1.bar(x + width/2, y_train.flatten()[:10], width, label='Real', color='#ff7f0e')

ax1.set_ylabel('House Price ($)')
ax1.set_xlabel('House Number')
ax1.set_title('Model Guess vs Real House Price (Training Data)')
ax1.set_xticks(x)
ax1.set_xticklabels(range(1,11))
ax1.legend()

# Prevent overlapping numbers by rotating them
ax1.bar_label(rects1_a, padding=5, rotation=90)
ax1.bar_label(rects1_b, padding=5, rotation=90)
ax1.set_ylim(0, max(max(train_guess.round().flatten()[:10]), max(y_train.flatten()[:10])) * 1.15)


# =========================================================================
# CHART 2: Right Panel (ax2)
# =========================================================================
rects2_a = ax2.bar(x - width/2, test_guess.round().flatten()[:10], width, label='Model', color='#2ca02c')
rects2_b = ax2.bar(x + width/2, y_test.flatten()[:10], width, label='Real', color='#d62728')

ax2.set_ylabel('House Price ($)')
ax2.set_xlabel('House Number')
ax2.set_title('Model Guess vs Real House Price (Testing Data)')
ax2.set_xticks(x)
ax2.set_xticklabels(range(1,11))
ax2.legend()

# Prevent overlapping numbers by rotating them
ax2.bar_label(rects2_a, padding=5, rotation=90)
ax2.bar_label(rects2_b, padding=5, rotation=90)
ax2.set_ylim(0, max(max(test_guess.round().flatten()[:10]), max(y_test.flatten()[:10])) * 1.15)


# 3. Clean up spacing and display
plt.tight_layout()
plt.show()