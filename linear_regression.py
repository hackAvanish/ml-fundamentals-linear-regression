import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame.sample(n=5000, random_state=42).reset_index(drop=True)

X = df.drop(columns=['MedHouseVal']).values
y = df['MedHouseVal'].values.reshape(-1, 1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Standardization
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Add intercept term
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# 2. Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# 3. Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X @ theta
        gradients = (1 / m) * (X.T @ (predictions - y))
        theta -= learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

# Train Custom Model
np.random.seed(42)
theta_init = np.random.randn(X_train_b.shape[1], 1)
theta_optimal, cost_history = gradient_descent(X_train_b, y_train, theta_init, lr=0.05, iterations=1000)

# Evaluate Custom Model
y_pred = X_test_b @ theta_optimal
test_mse = np.mean((y_pred - y_test) ** 2)
test_r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"Test MSE: {test_mse:.4f} | Test R²: {test_r2:.4f}")
