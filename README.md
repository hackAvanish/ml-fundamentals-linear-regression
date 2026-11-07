# Linear Regression from Scratch using NumPy

## Overview
This project implements a complete linear regression model from scratch using only **NumPy** and **Pandas**, avoiding high-level modeling libraries like `sklearn.linear_model`. 

## Mathematical Formulations

### 1. Hypothesis Function
The hypothesis function for linear regression with multiple features is represented in matrix notation as:
$$h_\theta(X) = X\theta$$
Where $X$ is the design matrix (including a column of ones for the intercept) and $\theta$ is the vector of parameters/weights.

### 2. Cost Function (Mean Squared Error - MSE)
The cost function measures the average squared difference between estimated values and the actual value:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### 3. Gradient Descent Update Rule
To minimize the cost, we update the parameters iteratively using gradients:
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
Where $\alpha$ is the learning rate and $m$ is the number of training samples.

## Evaluation & Results
* **Custom NumPy Model Test MSE:** 0.5234
* **Custom NumPy Model Test R²:** 0.6142
* **Scikit-Learn Benchmark Test MSE:** 0.5231
* **Scikit-Learn Benchmark Test R²:** 0.6145

The coefficients and metrics match closely, proving the correctness of the hand-built implementation.
