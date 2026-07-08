import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("ames2.csv")

X = df

X_train, X_test,= train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

X_train.to_csv("ames2_train.csv")
X_test.to_csv("ames2_test.csv")