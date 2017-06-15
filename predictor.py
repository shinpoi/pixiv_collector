import numpy as np
from chainer import Variable, serializers, using_config, no_backprop_mode
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

    # get original image !!
    def predict(self, img):
        with no_backprop_mode():
            xt = Variable(self.reduce_image(img))
            with using_config('train', False):
                yt = self.model.fwd(xt)
        ans = yt.data
        if np.argmax(ans[0]):
            return False, ans[0]
        else:
            return True, ans[0]

"""
f = EasyInceptionV2()
img_list = os.listdir('dataset/test_positive')
for img_name in img_list:
    print(f.predict(cv2.imread('dataset/test_positive/' + img_name, 0)))
"""
if __name__ == "__main__":
    import os
    import re
    import json
    from shutil import copyfile
    C = EasyInceptionV2()

    # read every images in dic and mov these to dic/po and dic/ne. and save value in value.json
    dic = "./test/"
    files = os.listdir(dic)
    is_image = re.compile('.*(\.jpg|\.gif|\.png|\.bmp)$', re.IGNORECASE)

    try:
        os.mkdir(dic + 'po')
        os.mkdir(dic + 'ne')
    except FileExistsError:
        print('File Exists!')

    dictionary = {}
    p = re.compile('[0-9]+_p[0-9]+')
    for file in files:
        if not is_image.match(file):
            continue
        image = cv2.imread(dic+file, 0)
        result, value = C.predict(image)
        print("image: %s, value: (%f, %f)" % (file, value[0], value[1]))

        img_name = p.search(file).group()
        dictionary[img_name] = '(%f, %f)' % (value[0], value[1])

        if result:
            copyfile(dic+file, dic+'po/'+file)
        else:
            copyfile(dic + file, dic + 'ne/' + file)

        with open(dic + 'value.json', 'w') as f:
            json.dump(dictionary, f)










