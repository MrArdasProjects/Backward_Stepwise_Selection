import pandas as pd
import numpy as np
from itertools import combinations

def fit_linear_regression(X, y):
    ones = np.ones((X.shape[0], 1))
    X_bias = np.hstack((ones, X))
    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y
    beta = np.linalg.inv(XtX) @ Xty
    return beta

def predict(X, beta):
    ones = np.ones((X.shape[0], 1))
    X_bias = np.hstack((ones, X))
    return X_bias @ beta

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_val_mse_manual(X, y, k=10):
    indices = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = len(indices) // k
    mse_list = []
    for i in range(k):
        val_idx = indices[i * fold_size:(i + 1) * fold_size if i != k - 1 else len(indices)]
        train_idx = np.setdiff1d(indices, val_idx)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        beta = fit_linear_regression(X_train, y_train)
        y_pred = predict(X_val, beta)
        mse = mean_squared_error(y_val, y_pred)
        mse_list.append(mse)
    return np.mean(mse_list)

def cross_val_mse_m0(y, k=10):
    indices = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = len(indices) // k
    mse_list = []
    for i in range(k):
        val_idx = indices[i * fold_size:(i + 1) * fold_size if i != k - 1 else len(indices)]
        train_idx = np.setdiff1d(indices, val_idx)
        y_train, y_val = y[train_idx], y[val_idx]
        y_pred = np.full_like(y_val, np.mean(y_train))
        mse = mean_squared_error(y_val, y_pred)
        mse_list.append(mse)
    return np.mean(mse_list)

def r2_score_manual(X, y):
    beta = fit_linear_regression(X, y)
    y_pred = predict(X, beta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def backward_stepwise_selection(X_df, y, initial_features):
    current_features = initial_features.copy()
    mse_list = []
    feature_sets = []

    for _ in range(len(initial_features)):
        X = X_df[current_features].values
        mse = cross_val_mse_manual(X, y)
        mse_list.append(mse)
        feature_sets.append(current_features.copy())

        if len(current_features) == 1:
            break

        worst_feature = None
        worst_r2 = None

        for feature in current_features:
            temp_features = current_features.copy()
            temp_features.remove(feature)
            X_temp = X_df[temp_features].values
            r2 = r2_score_manual(X_temp, y)
            if worst_r2 is None or r2 > worst_r2:
                worst_r2 = r2
                worst_feature = feature

        current_features.remove(worst_feature)

    mse_m0 = cross_val_mse_m0(y)
    mse_list.append(mse_m0)
    feature_sets.append([])

    return mse_list, feature_sets

def feature_importance(X_df, y_array):
    scores = {}
    for col in X_df.columns:
        X_col = X_df[[col]].values
        beta = fit_linear_regression(X_col, y_array)
        y_pred = predict(X_col, beta)
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        scores[col] = r2
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

df = pd.read_csv("Football_players_new.csv", encoding="ISO-8859-9")
features = ["Age", "Height", "Mental", "Skill"]
target = "Salary"

X_raw = df[features]
y = df[target].values
X_scaled = (X_raw - X_raw.mean()) / X_raw.std()

mse_list, feature_sets = backward_stepwise_selection(X_scaled, y, features)

print("--- MODEL RESULTS (M4 → M0, 10-fold CV) ---")
for i in range(len(mse_list)):
    print(f"Model M{len(features)-i} | MSE: {mse_list[i]:.2f} | Features: {feature_sets[i]}")

best_idx = np.argmin(mse_list)
print("\n--- MOST OPTIMAL MODEL ---")
print(f"Model: M{len(features)-best_idx} | Lowest MSE: {mse_list[best_idx]:.2f}")
print(f"Included Features: {feature_sets[best_idx]}")

print("\n--- FEATURE IMPORTANCE (Based on R² Score) ---")
importance = feature_importance(X_scaled, y)
for f, r2 in importance:
    print(f"{f} → R²: {r2:.3f}")

# This output confirms that all 5 models (M4 to M0) were evaluated.
# The model with the lowest cross-validation MSE is selected as most optimal.