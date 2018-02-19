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
import pandas as pd

from sklearn.model_selection import train_test_split

from learnpy.models.neighbors import KNeirestNeighbors


def first_example():
    df = pd.read_csv('iris.csv', header=0)
    df.columns = ['ft_1', 'ft_2', 'ft_3', 'ft_4', 'label']

    X = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = KNeirestNeighbors(k=5)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

if __name__ == '__main__':
    first_example()
