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


import operator

import numpy as np

from learnpy.support import Model


class KNeirestNeighbors(Model):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.dataset = X
        self.labels = y

    def predict(self, X):

        output = []
        X = np.atleast_2d(X)
        length = len(self.labels)

        for x in X:
            distances = []
            votes = {}

            for i in range(length):
                distance = np.sum( (x - self.dataset[i]) ** 2 ) ** 0.5
                distances.append([self.labels[i], distance])

            distances.sort(key=operator.itemgetter(1))

            for i in range(self.k):
                out = distances[i][0]
                if out in votes:
                    votes[out] += 1
                else:
                    votes[out] = 1

            votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)

            output.append(votes[0][0])

        return output


    def score(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        hypothesis = self.predict(X)
        correct = 0
        length = len(y)

        for i in range(length):
            if hypothesis[i] == y[i]:
                correct += 1

        return (correct / float(length)) * 100.0

