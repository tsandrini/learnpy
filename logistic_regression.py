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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from learnpy.models.linear_model import LogisticRegression, LogisticRegressionGD


def first_example():
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    df = pd.read_csv('data.csv', header=0)
    df.columns = ['grade1', 'grade2', 'label']

    X = df[['grade1', 'grade2']]
    X = np.array(X)
    X = min_max_scaler.fit_transform(X)

    y = df['label'].map(lambda x: float(x.rstrip(';')))
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegressionGD(
        learning_rate=0.005,
        iterations=100,
        tolerance=1e-5,
        fit_intercept=False
    )
    clf.fit(X_train, y_train)
    print("Logistic regression with Gradient descent optimalization:")
    print("Score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coefs)
    print("-----")

    clf = SKLogisticRegression()
    clf.fit(X_train, y_train)
    print("SKlearn logistic regression")
    print("Score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coef_)
    print("-----")

    clf = LogisticRegression(
        penalty='l2',
        learning_rate=0.001,
        iterations=15,
        tolerance=1e-10,
        fit_intercept=False
    )
    clf.fit(X_train, y_train)
    print("Logistic regression with Newton's optimalization:")
    print("Score: {0}".format(clf.score(X_test, y_test)))
    print(clf.coefs)
    print("-----")


def second_example():
    np.random.seed(12)
    num_observations = 5000
    iterations = 1000
    tolerance = 1e-5
    learning_rate = 0.001

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    X = np.vstack((x1)).astype(np.float64)
    y = np.hstack((np.zeros(num_observations),
                                np.ones(num_observations)))

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    clf = LogisticRegression(learning_rate=learning_rate, iterations=iterations, tolerance=tolerance)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(clf.coefs)

    skclf = SKLogisticRegression()
    skclf.fit(X_train, y_train)
    print(skclf.score(X_test, y_test))
    print(skclf.coef_)



if __name__ == '__main__':
    first_example()
