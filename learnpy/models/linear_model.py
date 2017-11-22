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

from numpy.linalg import inv
from learnpy.support import Model
from learnpy.runners import GradientDescentRunner
from learnpy.functions import coefficient_of_determination, sigmoid


class LinearMixin(object):

    def parse_X(self, X):
        X = np.atleast_1d(X)
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))

        return X


class LinearRegression(LinearMixin, Model):

    def __init__(self, fit_intercept=True, sample_weights=None):
        super(LinearRegression).__init__()
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = self.parse_X(X)

        X_T = X.T
        self.coefs = inv(X_T.dot(X)).dot(X_T).dot(y)

        return self

    def predict(self, X):
        X = self.parse_X(X)

        return np.dot(X, self.coefs)

    def score(self, X, y):
        return coefficient_of_determination(y, self.predict(X))

class LinearRegressionGD(LinearRegression, LinearMixin, Model):

    def __init__(self, learning_rate=0.001, iterations=1000, tolerance=0.0001, fit_intercept=True, sample_weights=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.coefs = sample_weights

    def fit(self, X, y, sample_weights=None):
        X = self.parse_X(X)

        if sample_weights:
            coefs = sample_weights
        elif not self.coefs:
            coefs = np.zeros(X.shape[1])
        else:
            coefs = self.coefs


        length = len(y)

        for i in range(self.iterations):
            last_coefs = coefs.copy()
            hypothesis = np.dot(X, coefs)

            error = hypothesis - y
            gradient = np.dot(X.T, error) / length
            coefs -= gradient * self.learning_rate

            if np.mean(np.abs(last_coefs - coefs)) < self.tolerance:
                break

        self.coefs = coefs

        return self

class LogisticRegression(Model):

    def __init__(self, learning_rate=0.001, iterations=100, tolerance=0.0001, fit_intercept=False, sample_weights=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.add_intercept = add_intercept
        self.coefs = sample_weights

    def fit(self, X, y, sample_weights=None):
        if self.add_intercept:
            intercept = np.ones(X.shape[0], 1)
            X = np.hstack(intercept, X)

        if sample_weights:
            coefs = sample_weights
        elif not self.coefs:
            coefs = np.zeros(X.shape[1])
        else:
            coefs = self.coefs


        for i in range(self.iterations):
            last_coefs = coefs.copy()
            scores = np.dot(X, coefs)
            hypothesis = sigmoid(scores)

            error = y - hypothesis
            gradient = np.dot(X.T, error)
            coefs += gradient * self.learning_rate

            if np.mean(np.abs(last_coefs - coefs)) < self.tolerance:
                break

        self.coefs = coefs

        return self

    def predict(self, X):
        scores = np.dot(X, self.coefs)
        hypothesis = sigmoid(scores)

        return np.round(hypothesis)

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)
