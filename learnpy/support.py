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


class Model(object):

    def __init__(self, *args, **kwargs):
        super(Model).__init__(args, kwargs)

    def fit(self, X, y):
        raise NotImplementedError(str(type(self)) + " does not implement fit")

    def predict(self, X, y):
        raise NotImplementedError(str(type(self)) + " does not implement predict")

    def score(self, X, y):
        raise NotImplementedError(str(type(self)) + " does not implement score")

    def __getstate__(self):
        d = dict()
        fields_to_del = getattr(self, 'fields_to_del', set())
        fields_to_keep = set(self.__dict__.keys()).difference(fields_to_)
        for name in fields_to_keep:
            d[name] = self.__dict__[name]

        return d
