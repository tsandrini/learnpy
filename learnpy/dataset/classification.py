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


def classify_spiral_data(num_samples, noise, radius=5, include_x2=True):
    X = []
    y = []
    n = num_samples / 2

    for label, delta in zip((1, 0), (0, np.pi)):
        for i in range(int(n)):
            _x = []
            r = i / n * radius # radius
            t = (1.75 * (i / n) * 2 * np.pi) + delta

            _x.append((r * np.sin(t))+ np.random.uniform(-1, 1) * noise)

            if include_x2 == True:
                _x.append((r * np.cos(t))+ np.random.uniform(-1, 1) * noise)

            X.append(_x)
            y.append([label])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def classify_circular_data(num_samples, noise, radius=5, include_x2=True):
    X = []
    y = []
    n = num_samples / 2

    for ratio in (0.5, 0.7):

        for i in range(int(n)):
            _x = []
            r = np.random.uniform(0, radius * ratio)
            angle = np.random.uniform(0, 2 * np.pi)

            _x1 = r * np.sin(angle)
            _x.append(_x1)

            _x2 = r * np.cos(angle)
            if include_x2 == True:
                _x.append(_x2)

            x1_noise = np.random.uniform(-radius, radius) * noise
            x2_noise = np.random.uniform(-radius, radius) * noise
            dist = np.sqrt((_x1 + x1_noise) ** 2 + (_x2 + x2_noise) ** 2)

            X.append(_x)
            y.append([1 if dist < (radius * 0.5) else 0])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def classify_XOR_data(num_samples, noise, radius=5, padding=0.3, include_x2=True):
    X = []
    y = []

    for i in range(num_samples):
        _x = []
        _x1 = np.random.uniform(-radius, radius)
        _x1 += padding if _x1 > 0 else -padding
        x1_noise = np.random.uniform(-radius, radius) * noise
        _x.append(_x1)

        _x2 = np.random.uniform(-radius, radius)
        _x2 += padding if _x2 > 0 else -padding
        x2_noise = np.random.uniform(-radius, radius) * noise
        if include_x2 == True:
            _x.append(_x2)

        y.append([1 if (_x1 + x1_noise) * (_x2 + x2_noise) >= 0 else 0])
        X.append(_x)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
