"""
 __                                       ____
/\ \                                     /\  _`\
\ \ \         __      __     _ __    ___ \ \ \L\ \ __  __
 \ \ \  __  /'__`\  /'__`\  /\`'__\/' _ `\\ \ ,__//\ \/\ \
  \ \ \L\ \/\  __/ /\ \L\.\_\ \ \/ /\ \/\ \\ \ \/ \ \ \_\ \
   \ \____/\ \____\\ \__/.\_\\ \_\ \ \_\ \_\\ \_\  \/`____ \
    \/___/  \/____/ \/__/\/_/ \/_/  \/_/\/_/ \/_/   `/___/> \
                                                       /\___/
                                                       \/__/
Created by Tomáš Sandrini
"""


import numpy as np


def sigmoid(X, derivative=False):
    if derivative == True:
        return np.exp(-(X)) / (1 + np.exp(-(X))) ** 2
    else:
        return 1 / (1 + np.exp(-(X)))

def logit(function, X):
    val = function(X)
    return np.log(val / 1 - val)

def step_gradient(X, y, thetas, alpha, hypothesis_function=None):
    _thetas = []
    length = len(y)
    hypothesis = np.dot(X, thetas)
    if callable(hypothesis_function):
        hypothesis = hypothesis_function(hypothesis)

    for i in range(len(thetas)):
        gradient = np.sum( (hypothesis - y) * X[:, 1] ) / length
        _thetas.append(thetas[i] - (gradient * alpha) / length)

    return _thetas

def squared_error(y_orig, y_line):
    return np.sum((y_orig - y_line) ** 2)

def coefficient_of_determination(y_orig, y_line):
    mean = np.mean(y_orig)
    y_mean_line = [mean for _ in y_orig]
    squared_error_reg = squared_error(y_orig, y_line)
    squared_error_mean = squared_error(y_orig, y_mean_line)

    return 1 - (squared_error_reg / squared_error_mean)
