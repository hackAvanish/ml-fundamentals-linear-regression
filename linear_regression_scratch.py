import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load and Prepare Dataset
print("Loading dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame
# Using a subset for faster demonstration, or use the whole dataset
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

X = df.drop(columns=['MedHouseVal']).values
y = df['MedHouseVal'].values.reshape(-1, 1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Standardization (mean=0, std=1)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Add intercept term (column of 1s) to X
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# 2. Define Hypothesis and Cost Function (MSE)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# 3. Gradient Descent Implementation
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X @ theta
        # Gradient of MSE cost function: (1/m) * X^T * (X*theta - y)
        gradients = (1 / m) * (X.T @ (predictions - y))
        theta -= learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history

# Initialize parameters
np.random.seed(42)
theta_init = np.random.randn(X_train_b.shape[1], 1)
lr = 0.05
n_iterations = 1000

print("Training model from scratch...")
theta_optimal, cost_history = gradient_descent(X_train_b, y_train, theta_init, lr, n_iterations)

# 4. Evaluate on Test Set (Manual MSE and R²)
y_pred = X_test_b @ theta_optimal

# Manual MSE
test_mse = np.mean((y_pred - y_test) ** 2)

# Manual R^2 Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
test_r2 = 1 - (ss_residual / ss_total)

print(f"\n--- Scratch Model Results ---")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R² Score: {test_r2:.4f}")

# 5. Compare with scikit-learn LinearRegression
skl_model = LinearRegression()
skl_model.fit(X_train_scaled, y_train)
skl_pred = skl_model.predict(X_test_scaled)

skl_mse = np.mean((skl_pred - y_test) ** 2)
skl_r2 = 1 - (np.sum((y_test - skl_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"\n--- Scikit-Learn Comparison ---")
print(f"Sklearn MSE: {skl_mse:.4f}")
print(f"Sklearn R² Score: {skl_r2:.4f}")

# 6. Plot Cost Convergence
plt.figure(figsize=(8, 5))
plt.plot(range(n_iterations), cost_history, color='b', lw=2)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (Cost)")
plt.title("Gradient Descent Cost Convergence")
plt.grid(True)
plt.show()
