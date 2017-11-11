"""
$$$$$$$\              $$\     $$\                           $$\      $$\ $$\
$$  __$$\             $$ |    $$ |                          $$$\    $$$ |$$ |
$$ |  $$ |$$\   $$\ $$$$$$\   $$$$$$$\   $$$$$$\  $$$$$$$\  $$$$\  $$$$ |$$ |
$$$$$$$  |$$ |  $$ |\_$$  _|  $$  __$$\ $$  __$$\ $$  __$$\ $$\$$\$$ $$ |$$ |
$$  ____/ $$ |  $$ |  $$ |    $$ |  $$ |$$ /  $$ |$$ |  $$ |$$ \$$$  $$ |$$ |
$$ |      $$ |  $$ |  $$ |$$\ $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |\$  /$$ |$$ |
$$ |      \$$$$$$$ |  \$$$$  |$$ |  $$ |\$$$$$$  |$$ |  $$ |$$ | \_/ $$ |$$$$$$$$\
\__|       \____$$ |   \____/ \__|  \__| \______/ \__|  \__|\__|     \__|\________|
          $$\   $$ |
          \$$$$$$  |
           \______/
Created by Tomáš Sandrini
"""


import numpy as np

from .decorators import time_usage


class LinearRegression():

    def __init__(self, iterations=None, learning_rate=None):
        self.thetas = []
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = self.validate_X(X)
        loss = []

        if not self.thetas:
            self.thetas = np.zeros(X.shape[1])

        for i in range(self.iterations):
            loss.append(self.loss(X, y))
            self.thetas = self.step_gradient(X, y)

        return loss

    def predict(self, X):
        X = self.validate_X(X)

        return np.dot(X, self.thetas)

    def loss(self, X, y):
        return (np.dot(X, self.thetas) - y) / (2 * len(y))

    def score(self, X, y):
        pct_error = np.mean(self.predict(X) / y)
        return 1 - pct_error if pct_error > 1 else pct_error

    def step_gradient(self, X, y):
        X = self.validate_X(X)
        thetas_out = []
        length = len(y)

        for i in range(len(self.thetas)):
            gradient = np.sum( (np.dot(X, self.thetas) - y) * X[:, i]) / length
            thetas_out.append(self.thetas[i] - (gradient * self.learning_rate) / length)

        return thetas_out

    @staticmethod
    def validate_X(X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        elif X[0][0] != 1:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        return X

def squared_error(y_orig, y_line):
    return np.sum((y_orig - y_line) ** 2)

def coefficient_of_determination(y_orig, y_line):
    mean = np.mean(y_orig)
    y_mean_line = [mean for _ in y_orig]
    squared_error_reg = squared_error(y_orig, y_line)
    squared_error_mean = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_mean)
