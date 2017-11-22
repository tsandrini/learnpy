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
import math
import quandl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import style
from sklearn import preprocessing
from learnpy.models.linear_model import LinearRegression, LinearRegressionGD
from sklearn.model_selection import train_test_split


def get_dataset(pcs, interval, features=1, step=2, correlation=False):
    i = 1
    X = []
    y = []
    for j in range(pcs):
        X.append([random.randrange(j, j * (k + 1) + 1) for k in range(features)])
        y.append(i + random.randrange(-interval, interval))
        if correlation and correlation == True:
            i += step
        else:
            i -= step

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def first_example():
    style.use('fivethirtyeight')
    X, y = get_dataset(1000, 500, 7, 3, True)
    pcs = math.ceil(0.8 * len(y))
    X = preprocessing.scale(X)
    X_train, y_train, X_test, y_test = X[:pcs], y[:pcs], X[pcs:], y[pcs:]

    iterations = 1000
    learning_rate = 0.01
    tolerance = 0.0001


    style.use('fivethirtyeight')
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    print("Classic LinerRegression score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coefs)

    clf = LinearRegressionGD(learning_rate=learning_rate, iterations=iterations, tolerance=tolerance)
    clf.fit(X_train, y_train)
    print("LinerRegressionGD score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coefs)


    # plt.plot(np.arange(max_iterations), loss)
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost function")
    # # plt.scatter(X_train, y_train, color='brown')
    # plt.show()

def second_example():
    df = quandl.get('WIKI/GOOGL')
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    forecast_col = 'Adj. Close'
    df.fillna(-9999, inplace=True)

    forecast_out = int(math.ceil(0.1 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    df.dropna(inplace=True)

    X = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    iterations = 1000
    learning_rate = 0.001
    tolerance = 0.0001


    style.use('fivethirtyeight')
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    print("Classic LinerRegression score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coefs)

    clf = LinearRegressionGD(learning_rate=learning_rate, iterations=iterations, tolerance=tolerance)
    clf.fit(X_train, y_train)
    print("LinerRegressionGD score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coefs)

    # plt.plot(np.arange(max_iterations), loss)
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost function")
    # plt.show()

if __name__ == '__main__':
    first_example()
