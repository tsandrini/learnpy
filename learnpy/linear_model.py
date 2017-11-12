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


class LinearRegression():
    """
    Linear Regression implementation

    Parameters
    ----------
    method: string required default 'matrix'
        determines which algorithm should be used
        avalaible ones are
        - matrix
        - gradient_descent

    max_iterations: integer optional default 100
        needed only while using Gradient Descent method
        determines how many times should be the gradient
        stepped while still not converging

    learning_rate: float optional
        needed only while using Gradient Descent method
        determines how much should be the coefficients
        changed every iteration

    tolerance: float optional default 0.0001
        needed only while using Gradient Descent method
        used to determine whether a given coefficient converges

    Attributes
    ----------
    coefs: array, shape(n_features, )
        Estimated coefficients for the Linear Regression problem

    intercept: float
        Independent term in the linear model
        Alias for coefs[0]
    """

    def __init__(self, *args, **kwargs):
        self.coefs = []
        method = kwargs['method'] if 'method' in kwargs else 'matrix'

        if method in set(['gradient', 'gradient_descent', 'descent']):
            try:
                self.method = 'gradient_descent'
                self.learning_rate = kwargs['learning_rate']
                self.max_iterations = kwargs['max_iterations'] if 'max_iterations' in kwargs else 100
                self.tolerance = kwargs['tolerance'] if 'tolerance' in kwargs else 0.0001
            except KeyError as e:
                print("While using the Gradient Descent method you have to specify 'learning_rate'")
        else:
            self.method = 'matrix'

    def fit(self, X, y):
        """
        Fit linear model

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples,]
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self: returns an instance of self
        """
        X = self.validate_X(X)

        if self.method == 'matrix':
            self.coefs = compute_coefs_by_matrix_multiplication(X, y)
        elif self.method == 'gradient_descent':
            self.coefs = gradient_descent(X, y, self.coefs, self.max_iterations, self.tolerance, self.learning_rate)

        return self

    def predict(self, X):
        """
        Predict target values by given data

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Data whose value will be predicted

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        X = self.validate_X(X)

        return np.dot(X, self.coefs)

    def score(self, X, y):
        """
        Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the
        residual sum of squares ((y_true - y_pred) ** 2).sum() and
        v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative
        (because the model can be arbitrarily worse). A constant model
        that always predicts the expected value of y, disregarding
        the input features, would get a R^2 score of 0.0.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Test samples

        y : numpy array of shape [n_samples]
            True values for given test samples

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        X = self.validate_X(X)
        return coefficient_of_determination(y, self.predict(X))

    @property
    def intercept(self):
        return self.coefs[0]

    @intercept.setter
    def intercept(self, value):
        self.coefs[0] = value

    @staticmethod
    def validate_X(X):
        X = np.atleast_1d(X)
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        elif X[0][0] != 1:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        return X

def compute_coefs_by_matrix_multiplication(X, y):
    X_transposed = X.transpose()
    return inv(X_transposed.dot(X)).dot(X_transposed).dot(y)

def step_gradient(X, y, coefs, learning_rate):
    coefs_out = []
    length = len(y)

    for i in range(len(coefs)):
        gradient = np.sum( (np.dot(X, coefs) - y) * X[:, i]) / length
        coefs_out.append(coefs[i] - (gradient * learning_rate) / length)

    return coefs_out

def gradient_descent(X, y, coefs, max_iterations, tolerance, learning_rate):
    if not coefs:
        coefs = np.zeros(X.shape[1])

    for i in range(max_iterations):
        last_coefs = coefs
        coefs = step_gradient(X, y, coefs, learning_rate)

        # Break if gradient converges with respect to the given tolerance
        if np.abs(np.mean([last_coefs[i] - coefs[i] for i in range(len(coefs))])) <= tolerance:
            break

    return coefs

def squared_error(y_orig, y_line):
    return np.sum((y_orig - y_line) ** 2)

def coefficient_of_determination(y_orig, y_line):
    mean = np.mean(y_orig)
    y_mean_line = [mean for _ in y_orig]
    squared_error_reg = squared_error(y_orig, y_line)
    squared_error_mean = squared_error(y_orig, y_mean_line)

    return 1 - (squared_error_reg / squared_error_mean)
