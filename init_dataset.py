# - coding: utf-8 -*-
# python 2.7

import cv2
import numpy as np
import os
import re
import logging
import time

# ./setting.py
import setting

# Parameters
SIZE = 133
LOG_DIR = setting.LOG_DIR
DATA_DIR = setting.DATA_DIR
SAVE_DIR = setting.DATA_DIR

# Crate directory
directory_lists = [LOG_DIR, SAVE_DIR]

for directory in directory_lists:
    try:
        os.mkdir(directory)
    except OSError:
        pass


# set logging
logging.basicConfig(level=setting.LOG_LEVEL,
                    format='[%(levelname)s]  \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    filename=LOG_DIR + 'init_' + time.strftime('%Y-%m-%d_%H-%M') + '.log',
                    filemode='a'
                    )

if setting.TO_CONSOLE:
    console = logging.StreamHandler()
    console.setLevel(setting.LOG_LEVEL)
    formatter = logging.Formatter('[%(levelname)s]  \t%(message)s\t')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

logging.info('Init start')


# Functions
# reduce image to size x size, blank will be filled with black(0x000000).
def reduce_image(filename):
    img = cv2.imread(filename, 0)
    img2 = np.zeros((1, SIZE, SIZE), dtype=np.float32)

    if img.shape[0] > img.shape[1]:
        img2[0, :SIZE, :int(img.shape[1] * SIZE / img.shape[0])] = cv2.resize(
            img[:, :], (int(img.shape[1] * SIZE / img.shape[0]), SIZE))
    else:
        img2[0, :int(img.shape[0] * SIZE / img.shape[1]), :SIZE] = cv2.resize(
            img[:, :], (SIZE, int(img.shape[0] * SIZE / img.shape[1])))
    img2 = img2 - img2.mean()
    return img2


# reduce and save image from src_dir to save_dir.
def create_training_data(src_dir, save_name=SAVE_DIR):
    is_image = re.compile('.*(\.jpg|\.gif|\.png|\.bmp)$', re.IGNORECASE)
    img_list = os.listdir(src_dir)

    count_sum = 0
    for img in img_list:
        flag = is_image.match(img)
        if flag:
            count_sum += 1
    if count_sum == 0:
        logging.error('Not found image from %s' % src_dir)
        return 0

    logging.info('Found %d image in "%s"' % (count_sum, src_dir))
    data = np.zeros((count_sum, 1, SIZE, SIZE), dtype=np.float32)
    count = 0
    for img in img_list:
        flag = is_image.match(img)
        if flag:
            img_re = reduce_image(src_dir + img)
            data[count] = img_re
            count += 1
            print("reduce image %d/%d " % (count, count_sum))

    np.save(save_name + '.npy', data)
    logging.info('Saved %d images from %s' % (count, src_dir))


create_training_data(DATA_DIR + 'train_positive/', SAVE_DIR + 'train_positive')
create_training_data(DATA_DIR + 'train_negative/', SAVE_DIR + 'train_negative')
create_training_data(DATA_DIR + 'test_positive/', SAVE_DIR + 'test_positive')
create_training_data(DATA_DIR + 'test_negative/', SAVE_DIR + 'test_negative')

logging.info('Init end')
