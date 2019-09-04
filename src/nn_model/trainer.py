# - coding: utf-8 -*-

import numpy as np
import logging
import conf

from chainer import using_config, no_backprop_mode, Variable, optimizers, serializers
from utils import random_target_list, mkdir

LOG = logging.getLogger('logger.nn_model.trainer')
GPU_DEVICE = 0


class Modelv2Trainer:
    def __init__(self, dataset, dataflag, model=conf.MODEL, use_gpu=conf.TRAIN_USE_GPU):
        # TODO: split data and training params
        self.model = model()
        self.optimizer = optimizers.Adam(alpha=conf.ADAM_RATE)
        self.optimizer.setup(self.model)

        self.dataset = dataset
        self.dataflag = dataflag

        self.dryrun = False

        if use_gpu:
            from chainer import cuda

            self.model.to_gpu(GPU_DEVICE)
            self.xp = cuda.cupy
            cuda.get_device_from_id(GPU_DEVICE).use()
            LOG.info('GPU used')
        else:
            self.xp = np

    @staticmethod
    def get_set_flag(data_g1, data_g2):
        len_g1 = data_g1.shape[0]
        len_g2 = data_g2.shape[0]

        data = np.zeros((len_g1 + len_g2, 1, conf.IMG_SIZE, conf.IMG_SIZE), dtype=np.float32)
        flag = np.zeros((len_g1 + len_g2, 2), dtype=np.float32)

        data[:len_g1] = data_g1
        data[len_g1:] = data_g2

        flag[:len_g1] = (1, 0)  # group 1
        flag[len_g1:] = (0, 1)  # group 2

        return data, flag

    def save_model(self, suffix=''):
        if self.dryrun:
            return
        # TODO: remove conf
        mkdir(conf.DATASET_DIR)
        model_path = conf.DATASET_DIR + '/cpu_model%s.npz' % suffix
        serializers.save_npz(model_path, self.model)
        LOG.info('Model Saved as: %s' % model_path)

    def train(self, epoch=3000, bc=64, eval_by=5):
        """

        :param epoch: max epoch
        :param bc: batch number
        :param eval_by: evaluate by each eval_by
        """
        dataset_len = len(self.dataset)  # number of train-data
        max_acc = 0
        max_acc_loop = 0
        max_acc_loss = 0
        model_num = 0

        for ep in range(epoch):
            LOG.debug('start epoch %d' % ep)

            train_data_index = random_target_list(dataset_len, 0.8)
            test_data_index = np.delete(np.arange(0, dataset_len, 1), train_data_index)
            n = len(train_data_index)  # len of train data

            train_data = self.dataset[train_data_index]
            test_data = self.dataset[test_data_index]

            train_flag = self.dataflag[train_data_index]
            test_flag = self.dataflag[test_data_index]

            # training
            shuffle_index = np.random.permutation(n)
            for i in range(0, n, bc):
                x = Variable(self.xp.array(train_data[shuffle_index[i: (i + bc) if (i + bc) < n else n], ]))
                y = Variable(self.xp.array(train_flag[shuffle_index[i: (i + bc) if (i + bc) < n else n], ]))
                self.model.cleargrads()
                loss = self.model(x, y)
                LOG.info("epoch: %d, loss = %f" % (ep, loss.data))
                loss.backward()
                self.optimizer.update()

            # evaluate
            if ep % eval_by == 0:
                with no_backprop_mode():
                    xt = Variable(test_data)
                    with using_config('train', False):
                        yt = self.model.fwd(xt)
                LOG.info("epoch: %d, loss = %f" % (ep, loss.data))
                self.model.cleargrads()
                ans = yt.data
                nrow, ncol = ans.shape
                tp = fp = fn = tn = 0

                # aggregate
                for i in range(nrow):
                    cls = int(np.argmax(ans[i]))
                    if test_flag[i][0]:  # y[i] == [1, 0]
                        if not cls:  # cls == 0
                            tp += 1
                        else:
                            fn += 1
                    else:  # y[i] == [0, 1]
                        if cls:  # cls == 1
                            tn += 1
                        else:
                            fp += 1

                acc = (tp + tn) * 1.0 / nrow
                LOG.info("epoch: %d, accuracy: %d/%d = %f" % (ep, (tp + tn), nrow, acc))
                LOG.info("epoch: %d, precision: %d/%d = %f" % (ep, tp, (tp + fp), (0 if tp == 0 else tp / (tp + fp))))
                LOG.info("epoch: %d, TrueNegative: %d/%d = %f" % (ep, tn, (fn + tn), (0 if tn == 0 else tn / (tn + fn))))
                LOG.info("epoch: %d, recall: %d/%d = %f" % (ep, tp, (tp + fn), tp / (tp + fn + 0.0000001)))

                if acc > max_acc:
                    max_acc = acc
                    max_acc_loop = ep
                    max_acc_loss = loss.data
                    if loss.data < conf.ACCEPT_LOSS:
                        if conf.SAVE_TEMP_MODEL:
                            self.save_model('_' + str(model_num))
                            model_num += 1

        LOG.info('Training End')
        LOG.info('MaxAcc=%f, loop=%d, loss=%f' % (max_acc, max_acc_loop, max_acc_loss))
        self.save_model()
        LOG.info('Model Saved')
