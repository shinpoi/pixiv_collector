from bs4 import BeautifulSoup
import re
import os
import setting
import time
import cv2
import json


# crate demo page by a file.
class DemoCreator(object):
    def __init__(self, image_path, date, log_name='', json_file='value.json'):
        self.root = setting.DEMO_ROOT
        self.template = setting.PAGE_TEMPLATE
        self.page_path = self.root + setting.PAGE_DIR
        self.image_path = image_path
        self.date = date
        self.log_name = log_name
        self.img_url = 'http://www.pixiv.net/member_illust.php?mode=medium&illust_id='
        self.illust_url = 'http://www.pixiv.net/ranking.php?mode=daily&content=illust&date='
        self.img_name = re.compile('[0-9]+_p[0-9]+')
        if json_file:
            try:
                with open(self.page_path + self.image_path + json_file, 'r') as f:
                    s = f.read()
                    self.value = json.loads(s)
            except FileNotFoundError:
                self.value = None
        else:
            self.value = None

    def create_rank_page(self):
        # open and parse template
        with open(self.template, 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'lxml')
        build_date = soup.find(id='build-date')
        day = soup.find(id='date')
        rank_link = soup.find(id='rank_link')

        is_image = re.compile('.*(\.jpg|\.gif|\.png|\.bmp)$', re.IGNORECASE)
        pattern_pid = re.compile('([0-9]+?)_p', re.IGNORECASE)

        po = soup.find(id='po')
        ne = soup.find(id='ne')
        pm = ['po/', 'ne/']
        switch = {pm[0]: po, pm[1]: ne}
        for x in pm:
            file = self.page_path + self.image_path + x
            img_list = os.listdir(file)
            for img in img_list:
                if is_image.match(img):
                    url = self.img_url + re.search(pattern_pid, img).group(1)
                    d = soup.new_tag('div', **{'class': 'show'})
                    a = soup.new_tag('a', href=url)
                    p = soup.new_tag('p')
                    if self.value:
                        # print(img)
                        name = self.img_name.search(img).group()
                        p.string = self.value[name]
                    else:
                        p.string = '[test_%s_001, test_%s_002]' % (x, x)
                    img = soup.new_tag('img', src='/'+setting.PAGE_DIR+self.image_path+x+img, **{'class': 'demo_image'})
                    a.append(img)
                    d.append(a)
                    d.append(p)
                    switch[x].append(d)

        day.string = self.date
        rank_link['href'] = self.illust_url + self.date
        build_date.string = time.strftime('%D-%M:%S')
        with open(self.page_path + self.date + '.html', 'w') as f:
            f.write(str(soup))

    def update_index(self):
        # log_dir = setting.LOG_DIR
        with open(self.root + 'index.html', 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'lxml')
        page = soup.find(id='rank_page')
        # log = soup.find(id='log_page')

        # page link
        a = soup.new_tag('a', href=('pixiv/' + self.date + '.html'))
        a.string = self.date[:4] + '-' + self.date[4:-2] + '-' + self.date[-2:]
        br = soup.new_tag('br')
        page.append(a)
        page.append(br)

        # log link
        """
        a = soup.new_tag('a', href='log/'+self.log_name)
        a.string = self.date[:4] + '-' + self.date[4:-2] + '-' + self.date[-2:]
        br = soup.new_tag('br')
        log.append(a)
        log.append(br)
        """
        with open(self.root + 'index.html', 'w') as f:
            f.write(str(soup))

    def load_value(self):
        pass


def reduce_img(date, group='/po/', d=''):
    if not d:
        d = setting.DEMO_ROOT + setting.PAGE_DIR + 'Daily_Rank_' + date + group
    pattern = re.compile('([0-9]+_p[0-9]).', re.IGNORECASE)
    img_list = os.listdir(d)
    n = 0
    for img in img_list:
        name = re.search(pattern, img).group(1)
        img = cv2.imread(d+img)
        if img.shape[0] < 400 and img.shape[1] < 400:
            continue
        if img.shape[0] > img.shape[1]:
            img = cv2.resize(img, (round(img.shape[1]*400/img.shape[0]), 400))
        else:
            img = cv2.resize(img, (400, round(img.shape[0]*400/img.shape[1])))
        cv2.imwrite(d+name+'.jpg', img)
        n += 1
        # print("%d/%d" % (n, len(img_list)))

# test
if __name__ == "__main__":
    t = DemoCreator(image_path='./co_test/', date='000')
    t.create_rank_page()
    pass
