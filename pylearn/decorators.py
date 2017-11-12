"""
      ___           ___           ___       ___           ___           ___           ___
     /\  \         |\__\         /\__\     /\  \         /\  \         /\  \         /\__\
    /::\  \        |:|  |       /:/  /    /::\  \       /::\  \       /::\  \       /::|  |
   /:/\:\  \       |:|  |      /:/  /    /:/\:\  \     /:/\:\  \     /:/\:\  \     /:|:|  |
  /::\~\:\  \      |:|__|__   /:/  /    /::\~\:\  \   /::\~\:\  \   /::\~\:\  \   /:/|:|  |__
 /:/\:\ \:\__\     /::::\__\ /:/__/    /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/ |:| /\__\
 \/__\:\/:/  /    /:/~~/~    \:\  \    \:\~\:\ \/__/ \/__\:\/:/  / \/_|::\/:/  / \/__|:|/:/  /
      \::/  /    /:/  /       \:\  \    \:\ \:\__\        \::/  /     |:|::/  /      |:/:/  /
       \/__/     \/__/         \:\  \    \:\ \/__/        /:/  /      |:|\/__/       |::/  /
                                \:\__\    \:\__\         /:/  /       |:|  |         /:/  /
                                 \/__/     \/__/         \/__/         \|__|         \/__/
Created by Tomáš Sandrini
"""


import time
import functools


def time_usage(func):
    """
    Prints time usage of a given function
    """
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print("elapsed time: %f" % (end_ts - beg_ts))
        return retval
    return wrapper

def trackcalls(func):
    """
    Checks whether a function has been called
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper
