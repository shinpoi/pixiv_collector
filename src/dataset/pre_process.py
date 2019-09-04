# - coding: utf-8 -*-

import cv2
import numpy as np
import conf

IMG_SIZE = conf.IMG_SIZE


def pre_process(img):
    # TODO: can handle rgb
    img = _resize(img)
    # TODO: retrain model by has hist_equal
    # img = _hist_equal(img.reshape((IMG_SIZE, IMG_SIZE)))
    return img


def _resize(img):
    """
    :param img: <np.array> shape(height, width) gray scale image

    :return: resized image
    """
    new_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # height > width
    if img.shape[0] > img.shape[1]:
        new_img[:IMG_SIZE, :int(img.shape[1] * IMG_SIZE / img.shape[0])] = \
            cv2.resize(img[:, :], (int(img.shape[1] * IMG_SIZE / img.shape[0]), IMG_SIZE))
    # width > height
    else:
        new_img[:int(img.shape[0] * IMG_SIZE / img.shape[1]), :IMG_SIZE] = \
            cv2.resize(img[:, :], (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
    return new_img


def _hist_equal(img):
    return cv2.equalizeHist(img)
