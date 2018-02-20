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


import os
import pickle


class Model(object):


    def __init__(self, *args, **kwargs):
        super(Model).__init__(args, kwargs)


    def fit(self, X, y):
        raise NotImplementedError(str(type(self)) + " does not implement fit")


    def predict(self, X, y):
        raise NotImplementedError(str(type(self)) + " does not implement predict")


    def score(self, X, y):
        raise NotImplementedError(str(type(self)) + " does not implement score")


    def save(self, path):
        with open(path, 'wb') as stream:
            pickle.dump(self, stream, pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as stream:
            clf = pickle.load(stream)

        return clf

