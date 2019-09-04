# - coding: utf-8 -*-

import cv2
import numpy as np
import conf
from utils import random_target_list

NOISE_RATE = conf.NOISE_RATE


def random_noise(img_arr):
    """
    :param img_arr: <np.array> shape(index, height, width, channel)
    """
    noise_list = random_target_list(len(img_arr), NOISE_RATE)

    noise_arr = np.array([_add_rand_noise(img) for img in img_arr[noise_list]], dtype=np.uint8)
    img_arr = np.concatenate((img_arr, noise_arr), 0)
    return img_arr


def _add_rand_noise(img):
    noise_funcs = (_inverse_color, _random_noise, _flip)
    n = len(noise_funcs)
    has_noise = False
    for noise_func in noise_funcs:
        if np.random.random() > 0.5:
            img = noise_func(img)
            has_noise = True
    if not has_noise:
        img = noise_funcs[np.random.randint(n)](img)
    return img


def _inverse_color(img):
    # return 255-img
    return -img


def _random_noise(img, rate=0.1):
    shape = img.shape
    img = img.reshape(-1)
    length = len(img)
    n = int(length * rate)

    rand_index = np.random.permutation(length)[:n]
    rand_value = np.random.randint(0, 255, n, dtype=np.uint8)
    img[rand_index] = rand_value

    return img.reshape(shape)


def _flip(img):
    return cv2.flip(img, 1)
