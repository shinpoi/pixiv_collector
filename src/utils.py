# - coding: utf-8 -*-

import os
import errno
import numpy as np


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def random_target_list(length, rate, sort=True):
    """
    :param length: <int> length of dataset
    :param rate: <float> rate of noise
    :param sort: <bool> sort result before return
    :return: random-chosen index list of dataset.
    """
    if length == 0 or length == 1 or rate == 0:
        return []
    n = round(length * rate)
    take_list = np.random.permutation(length)[:n]
    if sort:
        take_list.sort()
    return take_list
