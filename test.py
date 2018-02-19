#!/bin/env python3
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


import random

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from learnpy.dataset.regression import regress_plane
from learnpy.dataset.classification import classify_spiral_data, classify_circular_data, classify_XOR_data
# from learnpy.model_selection import train_test_split
from learnpy.models.neural_network import MLP
from learnpy.functions import relu, sigmoid, tanh, elu

from keras.datasets import cifar10

def main():
    # X, y = regress_plane(1000, 0.5, 100)
    # X, y = classify_spiral_data(2000, 1, 15)
    # X, y = classify_circular_data(3000, 0.2, 50)
    X, y = classify_XOR_data(2500, 0.8, 3, 0.8)
    X = preprocessing.scale(X)
    # y = preprocessing.scale(y)
    colors = ['#FF8000' if _y < 0 else '#009900' for _y in y]
    style.use('fivethirtyeight')

    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def first_example():
    # np.random.seed(0)
    X, y = classify_spiral_data(900, 3, 25)
    # X, y = classify_circular_data(600, 0.1, 25)
    # X, y = classify_XOR_data(600, 1, 8, 0.15)

    x1_squared = np.array([[_x[0] ** 2] for _x in X])
    X = np.append(X, x1_squared, axis=1)

    x2_squared = np.array([[_x[1] ** 2] for _x in X])
    X = np.append(X, x2_squared, axis=1)

    x1_x2 = np.array([[_x[0] * _x[1]] for _x in X])
    X = np.append(X, x1_x2, axis=1)

    sinx1 = np.array([[np.sin(_x[0])] for _x in X])
    X = np.append(X, sinx1, axis=1)

    sinx2 = np.array([[np.sin(_x[1])] for _x in X])
    X = np.append(X, sinx2, axis=1)


    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    colors = ['#FFC7A0' if _y <= 0 else '#AEEAA4' for _y in y_train]

    clf = MLP(layers=(X.shape[1], 8, 6, 2), activation=elu, has_biases=False)
    clf.fit(X_train, y_train, epochs=100, learning_rate=0.03, tolerance=10e-11, regularization='l1', regularization_rate=0.1)

    print()
    print("MLP({layers: %s, activation=%s, neurons=%s})" % (str(clf.layers), str(clf.activation), str(clf.num_neurons)))
    print("MLP score: " + str(clf.score(X_test, y_test)))

    plt.scatter(X_train[:, 0], X_train[:, 1], color=colors)

    y_predict = [np.argmax(_y) for _y in clf.predict(X_test)]
    colors = ['#FF6800' if _y <= 0 else '#20E800' for _y in y_predict]
    print(y_predict)

    plt.scatter(X_test[:, 0], X_test[:, 1], color=colors)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def second_example():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(len(y_test))

if __name__ == '__main__':
    second_example()
