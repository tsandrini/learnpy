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

import numpy as np

from random import shuffle

# from keras.datasets import cifar10

from learnpy.models.neural_network import SimpleNeuralNetwork


def first_example():
    # shape=(samples, 3, 32, 32)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    layers = (32 * 32, 20, 20, 10)

    clf = MLPClassifier(layers=layers)
    print(clf.weights)

def second_example():
    X = np.array([
        [3, 0],
        [2, 1],
        [2, 1],
        [1, 2],
        [2, 1],
        [1, 2],
        [1, 2],
        [0, 3],
    ])

    y = np.array([
        [1],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [1]
    ])

    np.random.seed(1)

    clf = SimpleNeuralNetwork((2, 3, 2))
    clf.fit(X, y, epochs=100, learning_rate=0.1, tolerance=0.1)
    print(clf.score(X, y))

if __name__ == '__main__':
    second_example()
