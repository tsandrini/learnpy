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


def regress_plane(num_samples, noise, radius=6, include_x2=True, y_range=(-1, 1)):
    X = []
    y = []

    for i in range(num_samples):
        _x = []
        _x1 = np.random.uniform(-radius, radius)
        x1_noise = np.random.uniform(-radius, radius) * noise
        _x.append(_x1)

        _x2 = np.random.uniform(-radius, radius)
        x2_noise = np.random.uniform(-radius, radius) * noise

        if include_x2 == True:
            _x.append(_x2)

        z = _x1 + x1_noise + _x2 + x2_noise
        amplitude = 4 * radius
        portion = (z + amplitude) / (2 * amplitude)

        y.append(y_range[0] + (y_range[1] - y_range[0]) * portion)
        X.append(_x)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
