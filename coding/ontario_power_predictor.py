import pandas as pd
from neural_network import activations 
from neural_network import loss_functions 
from neural_network import layers
import numpy as np

demand_train = pd.read_csv("data/ontario_power/train.csv")
demand_test = pd.read_csv("data/ontario_power/test.csv")

print(demand_train.head())
print(demand_test.head())

