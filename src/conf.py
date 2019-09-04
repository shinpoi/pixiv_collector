# - coding: utf-8 -*-

import logging
import time

import nn_model.model as model
from utils import mkdir

# -------------------------- #
# Crawler

# Pixiv account & password
P_ID = "----------"
P_PW = "----------"

# Directory to save image from crawler.py
CRAWLER_OUT_DIR = 'pixiv'

# -------------------------- #
# Log

# Directory to save log files.
LOG_DIR = 'logs'

# NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = logging.INFO

# env it's true, log will still be write into files.
TO_CONSOLE = True

mkdir(LOG_DIR)
logging.basicConfig(level=LOG_LEVEL,
                    format='[%(levelname)s]  \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    filename=LOG_DIR + '/' + time.strftime('%Y-%m-%d_%H-%M') + '.log',
                    filemode='a'
                    )

if TO_CONSOLE:
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('[%(levelname)s]  \t%(message)s\t')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# -------------------------- #
# Model

# Directory includes data-set.
DATASET_DIR = 'dataset'

# model params
TRAIN_USE_GPU = False
EVAL_USE_GPU = False
SAVE_TEMP_MODEL = True
MODEL_DATA_NAME = DATASET_DIR + '/model.npz'

NOISE_RATE = 0.5
ADAM_RATE = 0.00005
ACCEPT_LOSS = 0.9

MODEL = model.Modelv2
IMG_SIZE = 133

# -------------------------- #
# Demo Creator
CREATE_DEMO = False
DEMO_ROOT = '/var/www/pc_test/'
PAGE_DIR = 'pixiv'
PAGE_TEMPLATE = 'src/template.html'
