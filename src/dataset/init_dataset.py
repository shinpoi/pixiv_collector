# - coding: utf-8 -*-

import cv2
import numpy as np
import os
import re
import logging

from dataset.noise import random_noise
from dataset.pre_process import pre_process
from utils import mkdir

import conf

# Parameters
LOG = logging.getLogger('logger.dataset.init_dataset')
IMG_SIZE = conf.IMG_SIZE


# resize and save image from src_dir to save_dir.
def imgs2dataset(src_dir, save_name, test=False, debug=False, out_to_npy=True, out_dir=conf.DATASET_DIR):
    """
    convert images to np.array shape(index, height, width, channel) which can be used for model directly.

    :param src_dir: source images
    :param save_name: dataset file name. can be like folder/file
    :param test: test mode (do not add noise)
    :param debug: debug mode (out put add converted image to file)
    :param out_to_npy: dose save as a npy file
    :param out_dir: (debug and npy) out put dir

    :return: <np.array> shape(index, channel, height, width)
    """
    # TODO: cv2.imread not support gif
    is_image = re.compile('.*(\.jpg|\.png|\.bmp|\.jpeg)$', re.IGNORECASE)
    file_list = os.listdir(src_dir)

    _file_list = []
    imgs = []
    for img_name in file_list:
        if is_image.match(img_name):
            # TODO: handle rgb
            LOG.debug('read: ' + src_dir + '/' + img_name)
            img = cv2.imread(src_dir + '/' + img_name, 0)
            img = pre_process(img)
            imgs.append(img)
            _file_list.append(img_name)
    if not imgs:
        LOG.error('Not found image from %s' % src_dir)
        return 1

    imgs = np.array(imgs, dtype=np.uint8)
    LOG.info('Found %d image from "%s"' % (len(imgs), src_dir))
    if not test:
        imgs = random_noise(imgs)

    if debug:
        debug_dir = out_dir + '/debug/init_dataset'
        mkdir(debug_dir)
        for i in range(len(imgs)):
            cv2.imwrite(debug_dir + '/' + str(i) + '.jpg', imgs[i])

    # chainer need array: shape(index, channel, height, width)
    imgs = imgs.reshape((len(imgs), 1, IMG_SIZE, IMG_SIZE))
    if out_to_npy:
        mkdir(out_dir)
        np.save(out_dir + '/' + save_name + '.npy', imgs)
        LOG.info('Saved %d images from %s, the shape is: %s' % (len(imgs), src_dir, str(imgs.shape)))

    return imgs, _file_list
