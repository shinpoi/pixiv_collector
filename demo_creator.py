from bs4 import BeautifulSoup
import re
import os
import setting
import time


class DemoCreator(object):
    def __init__(self, root, image_path, date, log_name):
        self.root = root
        self.template = setting.PAGE_TEMPLATE
        self.page_path = self.root + setting.PAGE_DIR
        self.image_path = image_path
        self.date = date
        self.log_name = log_name
        self.img_url = 'http://www.pixiv.net/member_illust.php?mode=medium&illust_id='
        self.illust_url = 'http://www.pixiv.net/ranking.php?mode=daily&content=illust&date='

    def creat_rank_page(self):
        with open(self.template, 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'lxml')
        po = soup.find(id='po')
        ne = soup.find(id='ne')
        build_date = soup.find(id='build-date')
        day = soup.find(id='date')
        rank_link = soup.find(id='rank_link')

        positive_file = self.page_path + self.image_path + 'po/'
        negative_file = self.page_path + self.image_path + 'ne/'

        is_image = re.compile('.*(\.jpg|\.gif|\.png|\.bmp)$', re.IGNORECASE)
        pattern_pid = re.compile('([0-9]+?)_p', re.IGNORECASE)

        img_list = os.listdir(positive_file)
        for img in img_list:
            flag = is_image.match(img)
            if flag:
                url = self.img_url + re.search(pattern_pid, img).group(1)
                a = soup.new_tag('a', href=url)
                img = soup.new_tag('img', src='/'+setting.PAGE_DIR+self.image_path+'po/'+img, **{'class': 'demo_image'})
                a.append(img)
                po.append(a)

        img_list = os.listdir(negative_file)
        for img in img_list:
            flag = is_image.match(img)
            if flag:
                url = self.img_url + re.search(pattern_pid, img).group(1)
                a = soup.new_tag('a', href=url)
                img = soup.new_tag('img', src='/'+setting.PAGE_DIR+self.image_path+'ne/'+img, **{'class': 'demo_image'})
                a.append(img)
                ne.append(a)

        day.string = self.date
        rank_link['href'] = self.illust_url + self.date
        build_date.string = time.strftime('%D-%M:%S')
        with open(self.page_path + self.date + '.html', 'w') as f:
            f.write(str(soup))

    def update_index(self):
        log_dir = setting.LOG_DIR
        with open(self.root + 'index.html', 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'lxml')
        page = soup.find(id='rank_page')
        log = soup.find(id='log_page')

        # page link
        a = soup.new_tag('a', href=('pixiv/' + self.date + '.html'))
        a.string = self.date[:4] + '-' + self.date[4:-2] + '-' + self.date[-2:]
        br = soup.new_tag('br')
        page.append(a)
        page.append(br)

        # log link
        a = soup.new_tag('a', href='log/'+self.log_name)
        a.string = self.date[:4] + '-' + self.date[4:-2] + '-' + self.date[-2:]
        br = soup.new_tag('br')
        log.append(a)
        log.append(br)

        with open(self.root + 'index.html', 'w') as f:
            f.write(str(soup))


# test
"""
dc = DemoCreator(root=setting.ROOT, image_path='Daily_Rank_20170311/', date='20170311', log_name='Crawler_2017-03-12_19-50.log')
dc.creat_rank_page()
dc.update_index()
"""
