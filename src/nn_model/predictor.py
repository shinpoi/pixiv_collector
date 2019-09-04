import numpy as np
import cv2
import conf

from chainer import Variable, serializers, using_config, no_backprop_mode


class EasyInceptionV2:
    def __init__(self, size=None, model=None):
        self.size = conf.IMG_SIZE if not size else size
        self.model = conf.MODEL() if not model else model()
        serializers.load_npz('dataset/cpu_model.npz', self.model)

    def predict(self, img_arr):
        """

        :param img_arr: <np.array> shape(index, channel, height, width)
        :return: list of ( group, (score_A, score_B) )
        """
        img_arr = np.array(img_arr, dtype=np.float32)
        with no_backprop_mode():
            xt = Variable(img_arr)
            with using_config('train', False):
                yt = self.model.fwd(xt)
        ans = [(np.argmax(v), v) for v in yt.data]
        return ans


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
        image = cv2.imread(dic + file, 0)
        result, value = C.predict(image)
        print("image: %s, value: (%f, %f)" % (file, value[0], value[1]))

        img_name = p.search(file).group()
        dictionary[img_name] = '(%f, %f)' % (value[0], value[1])

        if result:
            copyfile(dic + file, dic + 'po/' + file)
        else:
            copyfile(dic + file, dic + 'ne/' + file)

        with open(dic + 'value.json', 'w') as f:
            json.dump(dictionary, f)
