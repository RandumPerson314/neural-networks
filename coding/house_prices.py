import numpy as np
import pandas as pd

from neural_network import activations
from neural_network import loss_functions
from neural_network import layers


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42

np.random.seed(SEED)

DROP_COLUMNS = [
    "SalePrice",
    "Order",
    "Unnamed: 0.1",
    "Unnamed: 0",
    "drop",
]


# --------------------------------------------------
# Load data
# --------------------------------------------------

df_train = pd.read_csv("data/ames2_train.csv")
df_test = pd.read_csv("data/ames2_test.csv")

for df in (df_train, df_test):
    df.replace({
        "True": 1,
        "False": 0,
        True: 1,
        False: 0,
    }, inplace=True)

feature_cols = [
    c for c in df_train.columns
    if c not in DROP_COLUMNS
]

X_train_df = df_train[feature_cols].apply(
    pd.to_numeric, errors="raise"
)
X_test_df = df_test[feature_cols].apply(
    pd.to_numeric, errors="raise"
)

y_train = df_train["SalePrice"].to_numpy(
    dtype=np.float64
).reshape(-1, 1)

y_test = df_test["SalePrice"].to_numpy(
    dtype=np.float64
).reshape(-1, 1)

X_train_raw = X_train_df.to_numpy(dtype=np.float64)
X_test_raw = X_test_df.to_numpy(dtype=np.float64)

if not (
    np.isfinite(X_train_raw).all()
    and np.isfinite(X_test_raw).all()
    and np.isfinite(y_train).all()
    and np.isfinite(y_test).all()
):
    raise ValueError("Data contains NaN or infinite values.")


# --------------------------------------------------
# Preprocess data
# --------------------------------------------------

# Remove features with zero variance in training.
raw_std = X_train_raw.std(axis=0)
keep = raw_std > 1e-12

removed_features = [
    name for name, retain in zip(feature_cols, keep)
    if not retain
]

X_train_raw = X_train_raw[:, keep]
X_test_raw = X_test_raw[:, keep]
feature_cols = [
    name for name, retain in zip(feature_cols, keep)
    if retain
]

# Standardize using training statistics only.
X_mean = X_train_raw.mean(axis=0)
X_std = X_train_raw.std(axis=0)

X_train = (X_train_raw - X_mean) / (X_std + 1e-8)
X_test = (X_test_raw - X_mean) / (X_std + 1e-8)

y_mean = y_train.mean()
y_std = y_train.std()

y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# Report unusual test inputs without flooding the terminal.
extreme_rows, extreme_features = np.where(np.abs(X_test) > 10)

print("=" * 52)
print("             AMES HOUSE PRICE MODEL")
print("=" * 52)
print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}")
print(f"Input features   : {X_train.shape[1]:,}")
print(f"Removed features : {len(removed_features):,}")

if len(extreme_rows):
    print(
        f"\nWarning: {len(extreme_rows)} standardized test "
        "value(s) exceed |10|."
    )
    for row, feature in zip(
        extreme_rows[:5], extreme_features[:5]
    ):
        print(
            f"  Row {row}, {feature_cols[feature]}: "
            f"{X_test[row, feature]:.3g}"
        )

    if len(extreme_rows) > 5:
        print(f"  ...and {len(extreme_rows) - 5} more.")

print("-" * 52)


# --------------------------------------------------
# Build neural network
# --------------------------------------------------

input_layer = layers.InputLayer(
    X_train.shape[1],
    32,
    activations.relu(),
)

hidden_layer = layers.HiddenLayer(
    32,
    10,
    activations.leaky_relu(),
)

output_layer = layers.OutputLayer(
    10,
    1,
    activations.linear(),
)


# --------------------------------------------------
# Training
# --------------------------------------------------

n_samples = len(X_train)

print("Training started")
print("-" * 52)

for epoch in range(1, EPOCHS + 1):
    indices = np.random.permutation(n_samples)

    X_shuffled = X_train[indices]
    y_shuffled = y_train_scaled[indices]

    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        input_fp = input_layer.forward_pass(X_batch)
        hidden_fp = hidden_layer.forward_pass(input_fp)
        output_layer.forward_pass(hidden_fp)

        output_layer.backward_pass(y_batch, LEARNING_RATE)
        hidden_layer.backward_pass(output_layer, LEARNING_RATE)
        input_layer.backward_pass(hidden_layer, LEARNING_RATE)

    if epoch == 1 or epoch % 100 == 0 or epoch == EPOCHS:
        train_pred_scaled = output_layer.forward_pass(
            hidden_layer.forward_pass(
                input_layer.forward_pass(X_train)
            )
        )

        train_loss = float(
            np.asarray(
                loss_functions.MSE(
                    train_pred_scaled,
                    y_train_scaled,
                )
            ).mean()
        )

        print(
            f"Epoch {epoch:4d}/{EPOCHS} "
            f"| Train MSE: {train_loss:.6f}"
        )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

train_pred_scaled = output_layer.forward_pass(
    hidden_layer.forward_pass(
        input_layer.forward_pass(X_train)
    )
)

test_pred_scaled = output_layer.forward_pass(
    hidden_layer.forward_pass(
        input_layer.forward_pass(X_test)
    )
)

if not (
    np.isfinite(train_pred_scaled).all()
    and np.isfinite(test_pred_scaled).all()
):
    raise ValueError("Model predictions contain NaN or infinity.")

train_pred = train_pred_scaled * y_std + y_mean
test_pred = test_pred_scaled * y_std + y_mean

train_mse = float(np.mean((train_pred_scaled - y_train_scaled) ** 2))
test_mse = float(np.mean((test_pred_scaled - y_test_scaled) ** 2))

train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

train_mae = float(np.mean(np.abs(train_pred - y_train)))
test_mae = float(np.mean(np.abs(test_pred - y_test)))


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 52)
print("                  FINAL RESULTS")
print("=" * 52)

print(f"{'Metric':<20}{'Train':>15}{'Test':>15}")
print("-" * 52)
print(f"{'Standardized MSE':<20}{train_mse:>15.6f}{test_mse:>15.6f}")
print(f"{'RMSE ($)':<20}{train_rmse:>15,.2f}{test_rmse:>15,.2f}")
print(f"{'MAE ($)':<20}{train_mae:>15,.2f}{test_mae:>15,.2f}")

print("\nSample predictions")
print("-" * 52)
print(f"{'Actual':>14}{'Predicted':>14}{'Error':>14}")
print("-" * 52)

for actual, predicted in zip(y_test[:10], test_pred[:10]):
    error = predicted[0] - actual[0]
    print(
        f"${actual[0]:>12,.0f}"
        f"${predicted[0]:>12,.0f}"
        f"${error:>12,.0f}"
    )

print("=" * 52)