import numpy as np
from chainer import Variable, serializers
import setting
import cv2
# import os


# Must include method predict() <- return a bool
class EasyInceptionV2():
    def __init__(self):
        self.size = setting.SIZE
        self.model = setting.MODEL()
        serializers.load_npz('cpu_model.npz', self.model)

    # Need np.array type's gray-scale image  | success: reduced image
    def reduce_image(self, img):
        img2 = np.zeros((self.size, self.size), dtype=np.float32)
        if img.shape[0] > img.shape[1]:
            img2[:, :int(img.shape[1] * self.size / img.shape[0])] = cv2.resize(img, (int(img.shape[1] * self.size / img.shape[0]), self.size))
        else:
            img2[:int(img.shape[0] * self.size / img.shape[1]), :] = cv2.resize(img, (self.size, int(img.shape[0] * self.size / img.shape[1])))
        img2 = img2 - img2.mean()
        return img2.reshape((1, 1, self.size, self.size))

    def predict(self, img):
        xt = Variable(self.reduce_image(img), volatile='on')
        yt = self.model.fwd(xt, test=True)
        ans = yt.data
        if np.argmax(ans[0]):
            return False
        else:
            return True

"""
f = EasyInceptionV2()
img_list = os.listdir('dataset/test_positive')
for img_name in img_list:
    print(f.predict(cv2.imread('dataset/test_positive/' + img_name, 0)))
"""


