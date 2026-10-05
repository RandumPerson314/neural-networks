import numpy as np
import pandas as pd
from neural_network import activations 
from neural_network import loss_functions 
from neural_network import layers

df_train = pd.read_csv("coding/data/ames2_train.csv")
df_test = pd.read_csv("coding/data/ames2_test.csv")

df_train = df_train.replace({
    "True": 1,
    "False": 0
})

df_test = df_test.replace({
    "True": 1,
    "False": 0
})

y_train = df_train["SalePrice"].to_numpy(dtype=np.float64).reshape(-1, 1)
X_train = df_train.drop(["SalePrice", "Order", "Unnamed: 0.1", "Unnamed: 0", "drop"], axis=1).to_numpy(dtype=np.float64)

y_train_mean = y_train.mean()
y_train_std = y_train.std()
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

y_train_scaled = (y_train - y_train_mean) / y_train_std 
X_train = (X_train - X_mean) / (X_std + 1e-8)

y_test = df_test["SalePrice"].to_numpy(dtype=np.float64).reshape(-1, 1)
X_test = df_test.drop(["SalePrice", "Order", "Unnamed: 0.1", "Unnamed: 0", "drop"], axis=1).to_numpy(dtype=np.float64)

y_test_scaled = (y_test - y_train_mean) / y_train_std 
X_test = (X_test - X_mean) / (X_std + 1e-8)
X_test = np.delete(X_test, [237,473, 472], axis=0)


print("largest test values per feature:")
print(np.max(np.abs(X_test), axis=0))

print("train X range:")
print(X_train.min(), X_train.max())

print("test X range:")
print(X_test.min(), X_test.max())

bad_feature = np.argmax(np.max(np.abs(X_test), axis=0))

print("bad feature index:", bad_feature)
print("bad feature values:")
print(X_test[:, bad_feature][np.abs(X_test[:, bad_feature]) > 100])

bad_rows = np.where(np.abs(X_test[:, bad_feature]) > 100)[0]

print("bad rows:")
print(bad_rows)     

# print(f"start with {X_train}")
# print(f"we want {y_train_scaled}")

input_layer_input = X_train
input_layer = layers.InputLayer(np.shape(input_layer_input)[1], 32, activations.relu())

hidden_layer_1 = layers.HiddenLayer(32, 10, activations.leaky_relu())

output_layer_target = y_train_scaled
output_layer = layers.OutputLayer(10, output_layer_target.shape[1], activations.linear())

epochs = 1000
batch_size = 64
lr = 0.01

n_samples = X_train.shape[0]

for epoch in range(epochs):

    indices = np.random.permutation(n_samples)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train_scaled[indices]

    for start in range(0, n_samples, batch_size):

        end = start + batch_size

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        input_fp = input_layer.forward_pass(X_batch)
        hidden_fp = hidden_layer_1.forward_pass(input_fp)
        output_fp = output_layer.forward_pass(hidden_fp)

        output_layer.backward_pass(y_batch, lr)
        hidden_layer_1.backward_pass(output_layer, lr)
        input_layer.backward_pass(hidden_layer_1, lr)

    if epoch % 100 == 0:
        train_pred = output_layer.forward_pass(
            hidden_layer_1.forward_pass(
                input_layer.forward_pass(X_train)
            )
        )

        print(
            epoch,
            "loss:",
            loss_functions.MSE(train_pred, y_train_scaled)
        )

print(f"last try {train_pred * y_train_std + y_train_mean}")
print(f"wanted: {y_train}")

input_layer_test_fp = input_layer.forward_pass(X_test)
hidden_layer_1_test_fp = hidden_layer_1.forward_pass(input_layer_test_fp)
output_layer_test_fp = output_layer.forward_pass(hidden_layer_1_test_fp)


test_pred = output_layer.forward_pass(
            hidden_layer_1.forward_pass(
                input_layer.forward_pass(X_test)
            )
    )

print("pred scaled:", test_pred[:10].flatten())
print("true scaled:", y_test_scaled[:10].flatten())
print("train predictions:")
print(train_pred.min(), train_pred.max())
print("test predictions:")
print(test_pred.min(), test_pred.max())
print("train:", loss_functions.MSE(train_pred, y_train_scaled))
print("test:", loss_functions.MSE(test_pred, y_test_scaled))