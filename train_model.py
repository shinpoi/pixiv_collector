# - coding: utf-8 -*-
# python 2.7

import numpy as np
from chainer import using_config, no_backprop_mode, Variable, optimizers, serializers, cuda
import time
import os
import logging
import setting
import src.model

# DEBUG
TEST_MODE = False

# Set logging
try:
    os.mkdir(setting.LOG_DIR)
except OSError:
    pass

logging.basicConfig(level=setting.LOG_LEVEL,
                    format='[%(levelname)s]   \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    filename=setting.LOG_DIR + 'train-model_' + time.strftime('%Y-%m-%d_%H-%M') + '.log',
                    filemode='a'
                    )

if setting.TO_CONSOLE:
    console = logging.StreamHandler()
    console.setLevel(setting.LOG_LEVEL)
    formatter = logging.Formatter('[%(levelname)s]  \t%(message)s\t')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# Read data
def read_data(mod='train'):
    logging.info('Read %s data Start' % mod)

    path_po = setting.DATA_DIR + mod + '_positive.npy'
    path_ne = setting.DATA_DIR + mod + '_negative.npy'

    data_po = np.load(path_po)
    data_ne = np.load(path_ne)

    data = np.zeros((data_po.shape[0] + data_ne.shape[0], 1, data_po.shape[2], data_po.shape[3]), dtype=np.float32)
    target = np.zeros((data_po.shape[0] + data_ne.shape[0], 2), dtype=np.float32)

    data[:data_po.shape[0]] = data_po
    data[data_po.shape[0]:] = data_ne

    target[:data_po.shape[0]] = (1, 0)  # positive
    target[data_po.shape[0]:] = (0, 1)  # negative

    return data, target


def save_model(modl, num=0, save=setting.SAVE_MODEL):
    if save:
        # serializers.save_npz('gpu_model.npz', model)
        serializers.save_npz('cpu_model_%d.npz' % num, modl)
        logging.critical('Model Saved as: cpu_model_%d.npz' % num)


# Split Training data and Test data
x_train, y_train = read_data('train')

if TEST_MODE:
    x_test, y_test = x_train, y_train
else:
    x_test, y_test = read_data('test')
    logging.info('Split end. Get %d train data and %d test data' % (y_train.shape[0], y_test.shape[0]))

# Model
model = src.model.CNN_02()
optimizer = optimizers.Adam(alpha=setting.ADAM_RATE)
optimizer.setup(model)

# Use GPU
if setting.GPU:
    gpu_device = 0
    cuda.get_device_from_id(gpu_device).use()
    model.to_gpu(gpu_device)
    xp = cuda.cupy

    # train test
    x_train = xp.array(x_train)
    y_train = xp.array(y_train)
    x_test = xp.array(x_test)

    logging.info('GPU used')

# Training
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
n = y_train.shape[0]  # number of train-data
bc = 60  # number of batch

MaxAcc = 0
MaxAcc_loop = 0
MaxAcc_loss = 0
model_num = 0

for j in range(201):
    print('start loop %d' % j)

    # training
    sff_index = np.random.permutation(n)
    for i in range(0, n, bc):
        x = Variable(x_train[sff_index[i: (i + bc) if (i + bc) < n else n], ])
        y = Variable(y_train[sff_index[i: (i + bc) if (i + bc) < n else n], ])
        model.cleargrads()
        loss = model(x, y)
        logging.debug("loop: %d, loss = %f" % (j, loss.data))
        loss.backward()
        optimizer.update()

    # evaluate
    if j % 5 == 0:
        with no_backprop_mode():
            xt = Variable(x_test)
            with using_config('train', False):
                yt = model.fwd(xt)
        logging.info("loop: %d, loss = %f" % (j, loss.data))
        model.cleargrads()
        ans = yt.data
        nrow, ncol = ans.shape
        ok = tp = fp = fn = tn = 0

        for i in range(nrow):
            cls = int(np.argmax(ans[i]))
            if y_test[i][0]:  # y[i] == [1, 0]
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
        logging.info("loop: %d, accuracy: %d/%d = %f" % (j, (tp + tn), nrow, acc))
        logging.info("loop: %d, precision: %d/%d = %f" % (j, tp, (tp + fp), tp / (tp + fp + 0.0000001)))
        logging.info("loop: %d, True Negative Rate: %d/%d = %f" % (j, tn, (fn + tn), tn / (fn + tn + 0.0000001)))
        logging.info("loop: %d, recall: %d/%d = %f" % (j, tp, (tp + fn), tp / (tp + fn + 0.0000001)))

        if acc > MaxAcc:
            MaxAcc = acc
            MaxAcc_loop = j
            MaxAcc_loss = loss.data
            if loss.data < setting.ACCEPT_LOSS:
                save_model(model, model_num)
                model_num += 1


logging.info('Training End')
logging.info('MaxAcc=%f, loop=%d, loss=%f' % (MaxAcc, MaxAcc_loop, MaxAcc_loss))

# Save Model
if setting.SAVE_MODEL:
    # serializers.save_npz('gpu_model.npz', model)
    model.to_cpu()
    serializers.save_npz('cpu_model_end.npz', model)
    logging.info('Model Saved')

# Restore Data (Debug)
"""
n = 0
for i in x_train[:5]:
    img = np.array((i*255)[0, :, :], dtype='uint8')
    cv2.imwrite(str(n)+'.jpg', img)
    n += 1

for i in x_test[:5]:
    img = np.array((i*255)[0, :, :], dtype='uint8')
    cv2.imwrite(str(n)+'.jpg', img)
    n += 1
"""
