from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Modelv1(Chain):
    """
    simple model. just for test.
    """

    def __init__(self):
        super(Modelv1, self).__init__(
            conv1=L.Convolution2D(1, 32, 9, stride=3),
            conv2=L.Convolution2D(32, 64, 5, stride=2),
            conv3=L.Convolution2D(64, 64, 3, stride=1),
            l4=L.Linear(None, 12),
            l5=L.Linear(12, 2),

            bn1=L.BatchNormalization(32),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        h1 = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 2)
        h2 = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h1))), 2)
        h3 = F.max_pooling_2d(F.relu(self.bn3(self.conv3(h2))), 2)
        h4 = F.dropout(F.softmax(self.l4(h3)))
        h5 = F.softmax(self.l5(h4))
        return h5


class Modelv2FULL(Chain):
    def __init__(self):
        super(Modelv2FULL, self).__init__(
            conv1=L.Convolution2D(1, 64, 7, stride=2, nobias=True),
            # MaxPool(3x3, 2)
            bn1=L.BatchNormalization(64),
            conv2a=L.Convolution2D(64, 64, 1, stride=1),
            conv2b=L.Convolution2D(64, 192, 3, stride=1),
            bn2=L.BatchNormalization(192),
            # MaxPool(3x3, 2)
            inc3a=L.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            # MaxPool(3x3, 2)
            inc4a=L.InceptionBN(320, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=L.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=L.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=L.InceptionBN(576, 32, 128, 192, 192, 256, 'avg', 128),
            inc5a=L.InceptionBN(608, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=L.InceptionBN(1024, 352, 192, 320, 192, 224, 'avg', 128),
            # AveragePool(7x7, 1)
            # Dropout(40%)
            out=L.Linear(1024, 2),
            # SoftMax
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        h = F.max_pooling_2d(self.bn1(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(self.bn2(self.conv2b(self.conv2a(h))), 3, stride=2)
        h = F.max_pooling_2d(self.inc3b(self.inc3a(h)), 3, stride=2)
        h = F.max_pooling_2d(self.inc4e(self.inc4d(self.inc4c(self.inc4b(self.inc4a(h))))), 3, stride=2)
        h = F.dropout(F.average_pooling_2d(self.inc5b(self.inc5a(h)), 7, stride=1), ratio=0.4)
        h = F.softmax(self.out(h))
        return h


class Modelv2(Chain):
    def __init__(self):
        super(Modelv2, self).__init__(
            conv1=L.Convolution2D(1, 64, 3, stride=1, nobias=True),
            # MaxPool(3x3, 2)
            bn1=L.BatchNormalization(64),
            conv2a=L.Convolution2D(64, 64, 1, stride=1),
            conv2b=L.Convolution2D(64, 128, 3, stride=1),
            bn2=L.BatchNormalization(128),
            # MaxPool(3x3, 2)
            inc3a=L.InceptionBN(128, 64, 48, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(256, 64, 64, 96, 64, 96, 'max', 64),
            # MaxPool(3x3, 2)
            inc4a=L.InceptionBN(320, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(576, 192, 128, 192, 192, 256, 'max', 128),
            inc5a=L.InceptionBN(768, 320, 192, 288, 192, 288, 'max', 128),
            # AveragePool(7x7, 1)
            # Dropout(40%)
            preout=L.Linear(1024, 64),
            # ReLu
            out=L.Linear(64, 2)
            # SoftMax
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        h = F.max_pooling_2d(self.bn1(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(self.bn2(self.conv2b(self.conv2a(h))), 3, stride=2)
        h = F.max_pooling_2d(self.inc3b(self.inc3a(h)), 3, stride=2)
        h = F.max_pooling_2d(self.inc4b(self.inc4a(h)), 3, stride=2)
        h = self.inc5a(h)
        h = F.dropout(F.average_pooling_2d(h, 7, stride=1), ratio=0.4)
        h = F.softmax(self.out(F.relu(self.preout(h))))
        return h
