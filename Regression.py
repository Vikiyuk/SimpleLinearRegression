# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:51:10 2024

@author: maiso
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
'''data = pd.read_csv("Salary.csv")
plt.figure(figsize=(10, 6))
plt.plot(data['YearsExperience'], data['Salary'], marker='o', linestyle='-')
plt.title('Observation')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()'''

import random
def initialize_coefficients() -> Tuple[float, float, float]:
    beta_0 = random.uniform(0, 1)
    beta_1 = random.uniform(0, 1)
    alpha = random.uniform(0, 1)
    return beta_0, beta_1, alpha
beta0, beta1, alpha = initialize_coefficients()
print("Współczynniki regresji:")
print("beta0 =", beta0)
print("beta1 =", beta1)
print("Alpha =", alpha)

def calculate_regression_function(x: np.ndarray, beta0: float, beta1: float) -> np.ndarray:
    f_x = beta0 + beta1 * x
    return f_x
x_values = np.array([1, 2, 3, 4, 5])
predicted_values = calculate_regression_function(x_values, beta0, beta1)
print("Predykcje funkcji regresji dla danych x:", predicted_values)


def calculate_error(predictions: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> float:
    m = len(y)
    SSR = 1 / (2 * m) * np.sum((predictions - y) ** 2)
    
    return SSR

y = np.array([2, 3, 4, 5, 6])

error = calculate_error(predicted_values, y, beta0, beta1)
print("Wartość błędu regresji:", error)

def calculate_gradient(predictions: np.ndarray, y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    m = len(y)
    diff = predictions - y
    gradient_beta0 = np.sum(diff) / m
    gradient_beta1 = np.sum(np.dot(diff, x)) / m
    return gradient_beta0, gradient_beta1
gradient_beta0, gradient_beta1 = calculate_gradient(predicted_values, y, x_values)
print("Gradient względem beta0:", gradient_beta0)
print("Gradient względem beta1:", gradient_beta1)

def update_regression_coefficients(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float, alpha: float) -> Tuple[float, float]:
    predictions = calculate_regression_function(x, beta0, beta1)
    gradient_beta0,gradient_beta1 = calculate_gradient(predictions, y, x)
    beta0 = beta0 - alpha  * gradient_beta0
    beta1 = beta1 - alpha * gradient_beta1
    return beta0,beta1