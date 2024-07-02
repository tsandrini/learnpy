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


def train_test_split(X, y, test_size=0.25, shuffle=True):
    dataset = np.array([[_x, _y] for _x, _y in zip(X, y)])

    if shuffle == True:
        np.random.shuffle(dataset)

    test_length = round(len(dataset) * test_size)
    X, y = dataset[:, 0], dataset[:, 1]

    return X[:test_length - 1], y[:test_length - 1], X[test_length:], y[test_length:]
