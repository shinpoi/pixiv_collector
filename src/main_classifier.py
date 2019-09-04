# - coding: utf-8 -*-

import logging
import sys
import os
from nn_model.predictor import EasyInceptionV2
from dataset import init_dataset
from utils import mkdir

LOG = logging.getLogger('logger.main_classifier')

if __name__ == '__main__':
    try:
        input_path = out_path = sys.argv[1]
    except IndexError as e:
        LOG.error('need input_path')
        raise e

    if input_path.endswith('/'):
        input_path = input_path[:-1]

    dataset, name_list = init_dataset.imgs2dataset(input_path, 'dataset',
                                                   test=True, debug=False, out_to_npy=False,
                                                   out_dir=out_path)

    p = EasyInceptionV2()
    ans = p.predict(dataset)
    LOG.info('fin evaluation')

    g1 = out_path + '/g1'
    g2 = out_path + '/g2'
    mkdir(g1)
    mkdir(g2)

    for i in range(len(ans)):
        if ans[i][0]:
            out = g1
        else:
            out = g2
        os.rename(input_path + '/' + name_list[i], out + '/' + name_list[i])

    LOG.info('fin')
