# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:51:10 2024

@author: maiso
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

#Zadanie 1
data = pd.read_csv("Salary.csv")
plt.figure(figsize=(10, 6))
plt.plot(data['YearsExperience'], data['Salary'], marker='o', linestyle='-')
plt.title('Observation')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

import random
def initialize_coefficients() -> Tuple[float, float, float]:
    beta_0 = random.uniform(0, 1)
    beta_1 = random.uniform(0, 1)
    alpha = random.uniform(0, 1)
    return beta_0, beta_1, alpha
def calculate_regression_function(x: np.ndarray, beta0: float, beta1: float) -> np.ndarray:
    f_x = beta0 + beta1 * x
    return f_x
def calculate_error(predictions: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> float:
    m = len(predictions)
    error = 1 / (2 * m) * np.sum((predictions - y) ** 2)
    return error

def calculate_gradient(predictions: np.ndarray, y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    m = len(y)
    gradient_beta0 = 1/m * np.sum(predictions - y)
    gradient_beta1 = np.mean((predictions - y) * x)
    return gradient_beta0, gradient_beta1


def update_regression_coefficients(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float, alpha: float) -> Tuple[float, float]:
    predictions = calculate_regression_function(x, beta0, beta1)
    gradient_beta0,gradient_beta1 = calculate_gradient(predictions, y, x)
    
    beta0 = beta0 - alpha  * gradient_beta0
    beta1 = beta1 - alpha * gradient_beta1
    return beta0,beta1

def normalize_data(x: np.ndarray) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def learn_and_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta0, beta1, alpha = initialize_coefficients()
    x = normalize_data(x)
    y = normalize_data(y)
    b0_values = []
    b1_values = []
    error_values = []
    allowed_error = 0.00001
    for epoch in range(1000):
        predicted_values = calculate_regression_function(x, beta0, beta1)
        error = calculate_error(predicted_values, y, beta0, beta1)
        gradient_beta0, gradient_beta1 = calculate_gradient(predicted_values, y, x)
        
        beta0,beta1 = update_regression_coefficients(x, y, beta0, beta1, alpha)
        
        
        b0_values.append(beta0)
        b1_values.append(beta1)
        error_values.append(error)
        if epoch > 0 and abs(error_values[-1] - error_values[-2]) < allowed_error:
            break
    return np.array(b0_values), np.array(b1_values), np.array(error_values)
import unittest
import pandas as pd

class SimpleLinearRegressionTest(unittest.TestCase):
    def test_initialize_coefficients(self):
        beta_0, beta_1, alpha = initialize_coefficients()
        
        self.assertTrue(0 < beta_0 < 1)
        self.assertTrue(0 < beta_1 < 1)
        self.assertTrue(0 < alpha < 1)
    def test_learn_and_fit(self):
        df = pd.read_csv('Salary.csv', sep=',')
        x = df['YearsExperience'].values.reshape(df['YearsExperience'].shape[0], 1)
        y = df['Salary'].values.reshape(df['Salary'].shape[0], 1)
        b0, b1, error = learn_and_fit(x, y)
        self.assertTrue(len(b0) > 1)
        self.assertTrue(len(b1) > 1)
        self.assertTrue(len(b0) == len(b1))
        self.assertTrue(all(i >= j for i, j in zip(error, error[1:]))) #Sprawdzenie, czy błędy nie rosną
        
unittest.main(argv=[''], verbosity=2, exit=False)
df = pd.read_csv('Salary.csv', sep=',')
x = df['YearsExperience'].values.reshape(df['YearsExperience'].shape[0], 1)
y = df['Salary'].values.reshape(df['Salary'].shape[0], 1)

b0, b1, error_values = learn_and_fit(x, y)
plt.plot(range(len(error_values)), error_values)
plt.title('Change in Regression Error')
plt.xlabel('Epoch')
plt.ylabel('Regression Error')
plt.grid(True)
plt.show()

num_epochs = len(b0)
epochs = [0, num_epochs // 2, num_epochs - 1]
for i, epoch_index in enumerate(epochs):
    x_unnormalized = data['YearsExperience']  
    x_norm = normalize_data(x_unnormalized)  
    predicted_values_norm = calculate_regression_function(x_norm, b0[epoch_index], b1[epoch_index])  # Predict with normalized data
    predicted_values = predicted_values_norm * (max(data['Salary']) - min(data['Salary'])) + min(data['Salary'])  # Denormalize predictions
    plt.figure(figsize=(10, 6))
    plt.plot(data['YearsExperience'], data['Salary'], marker='o', linestyle='-')
    plt.plot(x_unnormalized, predicted_values, label=f'Epoch {epoch_index}')
    plt.title(f'Observation with Regression Line (Epoch {epoch_index})')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.grid(True)
    plt.legend()
    plt.show()
def learn_and_fit2(x: np.ndarray, y: np.ndarray,alpha:float()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta0, beta1, alpha2 = initialize_coefficients()
    x = normalize_data(x)
    y = normalize_data(y)
    b0_values = []
    b1_values = []
    error_values = []
    allowed_error = 1
    for epoch in range(1000):
        predicted_values = calculate_regression_function(x, beta0, beta1)
        error = calculate_error(predicted_values, y, beta0, beta1)
        gradient_beta0, gradient_beta1 = calculate_gradient(predicted_values, y, x)
        
        beta0,beta1 = update_regression_coefficients(x, y, beta0, beta1, alpha)
        
        
        b0_values.append(beta0)
        b1_values.append(beta1)
        error_values.append(error)
        if epoch > 0 and abs(error_values[-1] - error_values[-2]) > allowed_error:
            break
    return np.array(b0_values), np.array(b1_values), np.array(error_values)
alpha_values = [0.01, 0.1, 0.5]  
desired_error = 0.01  
for alpha in alpha_values:
    b0, b1, error_values = learn_and_fit2(x, y, alpha)
    num_epochs = len(error_values)
    for epoch, error in enumerate(error_values):
        if error < desired_error:
            print(f"Dla alpha={alpha}, liczba epok dla bledu {desired_error}: {epoch + 1}")
            break
    else:
        print(f"Dla alpha={alpha}, trzeba wiecej niz {num_epochs} epoch.")