import numpy as np
import pandas as pd
from neural_network import activations 
from neural_network import loss_functions 
from neural_network import layers
import time

df_train = pd.read_csv("coding/data/ames2_train.csv")
df_test = pd.read_csv("coding/data/ames2_test.csv")

df_train = df_train.replace({
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


print(f"start with {X_train}")
print(f"we want {y_train_scaled}")

input_layer_input = X_train
input_layer = layers.InputLayer(np.shape(input_layer_input)[1], 32, activations.relu())
input_layer_fp = input_layer.forward_pass(input_layer_input)

hidden_layer_1 = layers.HiddenLayer(32, 10, activations.leaky_relu())
hidden_layer_1_fp = hidden_layer_1.forward_pass(input_layer_fp)

output_layer = layers.OutputLayer(10, y_train_scaled, activations.linear())
output_layer_fp = output_layer.forward_pass(hidden_layer_1_fp) 
print(f"first try: {output_layer_fp}")

for i in range(10000):
    lr = 0.0001
    output_layer_bp = output_layer.backward_pass(lr)
    hidden_layer_1_bp = hidden_layer_1.backward_pass(output_layer, lr)
    input_layer_bp = input_layer.backward_pass(hidden_layer_1, lr)
    input_layer_fp = input_layer.forward_pass(input_layer_input)
    hidden_layer_1_fp = hidden_layer_1.forward_pass(input_layer_fp)
    output_layer_fp = output_layer.forward_pass(hidden_layer_1_fp)
    if i % 2000 == 0:
        print(i)
        print("train:", loss_functions.MSE(output_layer_fp, y_train_scaled))
        print("test:", loss_functions.MSE(output_layer_fp, y_test_scaled))

print(f"last try {output_layer_fp   * y_train_std + y_train_mean}")

print(y_train)


