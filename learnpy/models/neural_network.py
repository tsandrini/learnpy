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


import pickle

import numpy as np

from random import shuffle

from learnpy.support import Model
from learnpy.functions import relu, quadratic_cost

class SimpleNeuralNetwork(Model):

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[1:], layers[:-1])]

    def predict(self, X):
        X = np.atleast_2d(X)
        y = []
        for x in X:
            a = x
            for w in self.weights:
                a = sigmoid(np.dot(w, a))
            y.append(a)

        return y

    def score(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        if y.shape[1] != 1 and self.layers[-1] != 1:
            y = [np.argmax(_y) for _y in y]

        results = [(np.argmax(self.predict(_x)), _y) for _x, _y in zip(X, y)]
        return sum(int(_x == _y) for _x, _y in results) / len(y)

    def fit(self, X, y, epochs=100, learning_rate=0.03, tolerance=0.001):

        X, y = self.parse_input_data(X, y)

        for j in range(0, epochs):

            last_cf_value = np.zeros(self.layers[-1])

            for _x, _y in zip(X, y):

                deriv_w = [np.zeros(w.shape) for w in self.weights]

                activations = [_x] # z vector which has been run through activation function
                zs = [] # z vectors (weight * previous_activation)

                # forward propagination
                for w in self.weights:
                    z = np.dot(w, activations[-1])
                    zs.append(z)
                    activations.append(sigmoid(z))

                cost = (activations[-1] - _y)

                deriv = cost * sigmoid(zs[-1], derivative=True)
                deriv_w[-1]= np.dot(deriv.reshape((deriv.shape[0], 1)), activations[-2].reshape((1, activations[-2].shape[0])))

                for i in range(2, self.num_layers):
                    deriv = np.dot(self.weights[-i + 1].T, deriv) * sigmoid(zs[-i], derivative=True)
                    deriv_w[-i] = np.dot(deriv.reshape((deriv.shape[0], 1)), activations[-i - 1].reshape((1, activations[-i - 1].shape[0])))

                self.weights = [w - dw * learning_rate for w, dw in zip(self.weights, deriv_w)]

                if np.sum(np.abs(cost - last_cf_value)) < tolerance:
                    print("Training stopped due to overfitting.")
                    break

            else:
                continue

            break

    def parse_input_data(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        if y.shape[1] == 1 and self.layers[-1] != 1:
            parsed_ys = []
            for _y in y:
                tmp = np.zeros(self.layers[-1])
                tmp[_y[0]] = 1
                parsed_ys.append(tmp)
            y = parsed_ys

        return X, y

class MLP(Model):

    def __init__(self, *args, **kwargs):
        if not 'initialize' in kwargs or kwargs['initialize'] == True:
            self.initialize(*args, **kwargs)


    def initialize(self, *args, **kwargs):
        if 'layers' not in kwargs:
            raise ValueError("You should provide 'layers' argument for MLP to work properly")

        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' else 0.3
        self.tolerance = kwargs['tolerance'] if 'tolerance' in kwargs else 10e-6

        self.regularization = kwargs['regularization'] if 'regularization' else None
        self.regularization_rate = kwargs['regularization_rate'] if 'regularization_rate' else 0.03

        self.has_biases = kwargs['has_biases'] if 'has_biases' in kwargs else True
        self.layers = kwargs['layers']
        self.num_layers = len(self.layers)
        self.activation = kwargs['activation'] if 'activation' in kwargs else relu
        self.cost_function = kwargs['cost_function'] if 'cost_function' in kwargs else quadratic_cost
        self.weights = np.array([np.random.randn(x, y) for x, y in zip(self.layers[1:], self.layers[:-1])])

        if self.has_biases == True:
            self.biases = np.array([np.random.rand(y) for y in self.layers[1:]])
        else:
            self.biases = None


    def predict(self, X):
        X = np.atleast_2d(X)
        y = []

        if self.has_biases == True:
            for x in X:
                a = x
                for w, b in zip(self.weights, self.biases):
                    a = self.activation(np.dot(w, a)) + b
                y.append(a)
        else:
            for x in X:
                a = x
                for w in self.weights:
                    a = self.activation(np.dot(w, a))
                y.append(a)

        return y


    def fit(self, X, y, epochs=100):
        X, y = self.parse_input_data(X, y)
        dataset = np.array([[_x, _y] for _x, _y in zip(X, y)])

        if self.has_biases == True:
            self.gradient_descent(dataset, epochs)
        else:
            self.gradient_descent_without_biases(dataset, epochs)


    def gradient_descent(self, dataset, epochs):
        for j in range(0, epochs):
            last_cf = np.zeros(self.layers[-1])
            np.random.shuffle(dataset)

            for _x, _y in dataset:

                deltas_w = [np.zeros(w.shape) for w in self.weights]
                deltas_b = [np.zeros(b.shape) for b in self.biases]

                activations = [_x]
                zs = []

                # Forward propagination
                for w, b in zip(self.weights, self.biases):
                    z = np.dot(w, activations[-1]) + b
                    zs.append(z)
                    activations.append(self.activation(z))

                squared_w = 0
                for w in self.weights:
                    squared_w += np.sum(w ** 2)

                cf = self.cost_function(activations[-1], _y) + 0.5 * self.regularization_rate * squared_w

                # Backpropagination
                delta = self.cost_function(activations[-1], _y, derivative=True) * self.activation(zs[-1], derivative=True)
                deltas_b[-1] = delta
                deltas_w[-1] = np.dot( delta.reshape( (delta.shape[0], 1) ), activations[-2].reshape( (1, activations[-2].shape[0]) ) )

                for i in range(2, self.num_layers):
                    delta = np.dot(self.weights[-i + 1].T, delta) * self.activation(zs[-i], derivative=True)
                    deltas_b[-i] = delta
                    deltas_w[-i] = np.dot( delta.reshape( (delta.shape[0], 1) ), activations[-i -1].reshape( (1, activations[-i -1].shape[0]) )  ) + self.weights[-i] * self.regularization_rate

                self.weights = [w - dw * self.learning_rate for w, dw in zip(self.weights, deltas_w)]
                self.biases = [b - bw * self.learning_rate for b, bw in zip(self.biases, deltas_b)]

                print("|cf - last_cf| = " + str(np.sum(np.abs(cf - last_cf))))
                if np.sum(np.abs(cf - last_cf)) < self.tolerance:
                    print("Training stopped due to overfitting.")
                    break

            else:
                continue

            break


    def gradient_descent_without_biases(self, dataset, epochs):
        for j in range(0, epochs):
            last_cf = np.zeros(self.layers[-1])
            np.random.shuffle(dataset)

            for _x, _y in dataset:

                deltas_w = [np.zeros(w.shape) for w in self.weights]

                activations = [_x]
                zs = []

                # Forward propagination
                for w in self.weights:
                    z = np.dot(w, activations[-1])
                    zs.append(z)
                    activations.append(self.activation(z))

                cf = self.cost_function(activations[-1], _y)

                # Backpropagination
                delta = self.cost_function(activations[-1], _y, derivative=True) * self.activation(zs[-1], derivative=True)
                deltas_w[-1] = np.dot( delta.reshape( (delta.shape[0], 1) ), activations[-2].reshape( (1, activations[-2].shape[0]) ) )

                for i in range(2, self.num_layers):
                    delta = np.dot(self.weights[-i + 1].T, delta) * self.activation(zs[-i], derivative=True)
                    deltas_w[-i] = np.dot( delta.reshape( (delta.shape[0], 1) ), activations[-i -1].reshape( (1, activations[-i -1].shape[0]) )  )

                self.weights = [w - dw * self.learning_rate for w, dw in zip(self.weights, deltas_w)]

                print("|cf - last_cf| = " + str(np.sum(np.abs(cf - last_cf))))
                if np.sum(np.abs(cf - last_cf)) < self.tolerance:
                    print("Training stopped due to overfitting.")
                    break

            else:
                continue

            break


    def parse_input_data(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        if y.shape[1] == 1 and self.layers[-1] != 1:
            parsed_ys = []
            for _y in y:
                tmp = np.zeros(self.layers[-1])
                tmp[int(_y[0])] = 1
                parsed_ys.append(tmp)
            y = parsed_ys

        return X, y


    def score(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        if y.shape[1] != 1 and self.layers[-1] != 1:
            y = [np.argmax(_y) for _y in y]

        results = [(np.argmax(self.predict(_x)), _y) for _x, _y in zip(X, y)]
        return sum(int(_x == _y) for _x, _y in results) / len(y)
