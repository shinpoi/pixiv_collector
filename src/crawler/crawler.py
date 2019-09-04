# -*- coding: utf-8 -*-
# TODO: disable after 2019. rewrite by headless browser like puppeteer

import time
import datetime
import logging
import re
import requests
import json
import urllib3
from bs4 import BeautifulSoup


import conf
from utils import mkdir

# Parameter
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ID = conf.P_ID
PW = conf.P_PW

LOG = logging.getLogger('logger.crawler.crawler')

SAVE_PATH = conf.CRAWLER_OUT_DIR
mkdir(SAVE_PATH)

_host = 'www.pixiv.net'
_homepage = 'https://www.pixiv.net/'

_logging_hp = 'https://accounts.pixiv.net'
_pre_login_page = _logging_hp + '/login?lang=en&source=pc&view_type=page&ref=wwwtop_accounts_index'
_login_api = _logging_hp + '/api/login?lang=zh'

_ranking_page = 'https://www.pixiv.net/ranking.php'

_image_page = 'https://www.pixiv.net/member_illust.php?mode=medium&illust_id='

_manga_page = 'https://www.pixiv.net/member_illust.php?mode=manga&illust_id='
_works_page = 'https://www.pixiv.net/member_illust.php?type=all&id='
_bookmarks_page = 'https://www.pixiv.net/bookmark.php?rest=show&id='
_score_api = 'https://www.pixiv.net/rpc_rating.php'


class Crawler(object):
    def __init__(self, user, pw, out_path=SAVE_PATH, date=DATE):
        self.id = user
        self.pw = pw
        self.uid = ''
        self.out_path = out_path
        self.times = date

        self.headers_base = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'Referer': _homepage,
        }
        self.ss = requests.session()
        self.ss.headers.update(self.headers_base)
        self.ss.verify = False

        self.pattern_host = re.compile('https?://(.+?)/')
        self.pattern_tt = re.compile('pixiv.context.token = "([a-zA-Z0-9]+?)";')
        self.pattern_pid = re.compile('pixiv.user.id = "([0-9]+)";')
        self.pattern_uid = re.compile('id=([0-9]+)')
        self.pattern_img_name = re.compile('[0-9]+_p[0-9]+')

        self.pattern_original_img = re.compile('"original":"(.+)?"')
        self.pattern_img_count = re.compile('"pageCount":([0-9]+)')
        self.load_cookies()

    def load_cookies(self):
        LOG.info('init cookies...')
        try:
            LOG.info('read cookies')
            with open(self.out_path + '/cookies.json', 'r') as f:
                self.ss.cookies = requests.utils.cookiejar_from_dict(json.loads(f.read()))
            if not self.is_login():
                self.login()
        except IOError or FileNotFoundError:
            LOG.warning("Didn't find cookies")
            self.login()

    def dump_cookies(self):
        cookies = requests.utils.dict_from_cookiejar(self.ss.cookies)
        with open(self.out_path + 'cookies.json', 'w') as f:
            f.write(json.dumps(cookies))
        LOG.info('cookies was saved')

    def save_file(self, name, bin_data):
        """
        save bin-data to file
        """
        if not bin_data:
            LOG.error("save_file(): Did't get data")
            return False
        with open(self.out_path + name, 'wb') as f:
            f.write(bin_data)
        LOG.info('File %s saved' % self.out_path + name)

    def is_login(self):
        """
        check is login by check is my-id in homepage
        """
        # Get Homepage
        LOG.info('access homepage')
        time.sleep(1)

        r = self.ss.get(_homepage, headers={'Host': _host})

        print('----------------------')
        print(r.content.decode('utf-8'))
        print('----------------------')

        self.ss.cookies = r.cookies
        check_id = self.id in r.content.decode('utf-8')
        if not check_id:
            LOG.error('Can not get uid from homepage')
            return False

        # Set uid
        self.uid = self.pattern_pid.search(r.content).group(1)
        self.dump_cookies()
        return True

    def login(self):
        # Pre-Login
        LOG.info('Login Pixiv ...')
        r = self.ss.get(_pre_login_page)
        self.ss.cookies = r.cookies
        LOG.info('Pre-Login... status_code: %s' % r.status_code)

        # Login
        headers = {'Referer': _pre_login_page, 'Host': _host, 'origin': _logging_hp}
        post_data = {}

        soup = BeautifulSoup(r.text, 'lxml')
        f = soup.find("form", action="/login")
        for i in f.find_all('input'):
            try:
                post_data[i['name']] = i['value']
            except KeyError:
                post_data[i['name']] = ''
        post_data['pixiv_id'] = self.id
        post_data['password'] = self.pw

        r = self.ss.post(_login_api, data=post_data, headers=headers)
        print('post data: ' + str(post_data))

        self.ss.cookies = r.cookies
        LOG.info('Login... status_code: %s' % r.status_code)
        time.sleep(1)
        if self.is_login():
            LOG.info("Login successful. ID = %s " % self.id)
        else:
            LOG.error("Login Failed! ID = %s " % self.id)
            raise RuntimeError

    @staticmethod
    def image_format(url):
        pattern_img_url = re.compile('/[0-9]+_p[0-9]+.(jpg|gif|png|bmp)')
        name = re.search(pattern_img_url, url).group(0)
        img_format = re.search(pattern_img_url, url).group(1)
        return name[1:], img_format

    def get_images_by_id(self, img_id, referer=None):
        # get url of original image
        url = _image_page + img_id
        headers = {'Host': _host}
        if referer:
            headers['Referer'] = referer

        # access medium image page
        r = self.ss.get(url, headers=headers)
        img_url = self.pattern_original_img.search(r.content)
        img_count = self.pattern_img_count.search(r.content)

        if not img_url or img_count:
            LOG.warning('can not get image message from %s' % url)
            return None

        img_url = img_url.group(1).replace('\\', '')
        img_count = int(img_count.group(1))

        headers = {'Referer': url}
        img_data_list = []
        for i in range(img_count):
            url = img_url.replace('_p0', ('_p%d' % i))
            name, img_format = self.image_format(url)
            # ignore gif
            if img_format == 'gif':
                LOG.debug('ignore gif image: ' + url)
                continue
            img = self.ss.get(url, headers=headers)
            img_data_list.append((name, img_format, img.content))
            time.sleep(0.5)
        return img_data_list

    # Add image-id of (daily) ranking into id-set. | Success: id-set ;
    def scan_ranking(self, mode='daily', content='illust', page=4, date='', id_set=None):
        if not date:
            date = self.times
        if not id_set:
            id_set = set()

        LOG.info('Start scan ranking.(mode=%s, content=%s, date=%s)' % (mode, content, date))

        overview_url = _ranking_page + '?mode=' + mode + '&content=' + content + '&date=' + date
        headers = {'Referer': _homepage, 'Host': _host}

        r = self.ss.get(overview_url, headers=headers)
        token = re.search(self.pattern_tt, r.text).group(1)
        headers['Referer'] = overview_url

        for i in range(page):
            para = '?mode=' + mode + '&content=' + content + '&date=' + date + '&p=' + str(
                i + 1) + '&format=json&tt=' + token
            json_url = _ranking_page + para
            json_rank = json.loads(self.ss.get(json_url, headers=headers).text)
            try:
                for img_json in json_rank['contents']:
                    id_set.add(str(img_json['illust_id']))
                time.sleep(0.5)
            except KeyError:
                print(json_rank)
                LOG.error('Get json of page %s failed' % str(i + 1))

        LOG.info('End scan ranking.')
        return id_set

    # Add artist's works/bookmarks image-id into id-set. | Success: id-set ;
    def scan_artist(self, uid, class_='works', id_set=None):
        if not id_set:
            id_set = set()
        LOG.info('Start scan user(id=%s), scan_type is: %s.' % (uid, class_))

        if class_ == 'works':
            page = _works_page
        elif class_ == 'bookmarks':
            page = _bookmarks_page
        else:
            LOG.error('scan_artist(): Get a wrong value of class_: %s' % class_)
            raise ValueError

        headers = {'Host': _host}
        max_page = '0'
        new_max_page = '1'

        while max_page != new_max_page:
            max_page = new_max_page
            url = page + uid + '&p=' + max_page
            r = self.ss.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'lxml')
            new_max_page = soup.find('ul', class_='page-list')
            if new_max_page:
                new_max_page = new_max_page.find_all('li')[-1].text
            else:
                new_max_page = '1'
                break

        LOG.info('Has %s pages!' % new_max_page)
        for i in range(int(max_page)):
            url = page + uid + '&p=' + str(i + 1)
            r = self.ss.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'lxml')
            tag_list = soup.find_all('img', class_='ui-scroll-view')
            for tag in tag_list:
                id_set.add(tag['data-id'])
            time.sleep(0.5)

        LOG.info('End scan user')
        return id_set

    # Save original image by id-set.
    def crawler(self, id_set, save_file):
        if not id_set:
            LOG.warning('not img to craw')
            return

        mkdir(save_file)
        mkdir(save_file + 'po/')
        mkdir(save_file + 'ne/')

        LOG.info('Start for write image in %s' % save_file)
        num = str(len(id_set))
        n = 0
        for img_id in id_set:
            n += 1
            LOG.info("check image %s/%s" % (str(n), num))
            images_p = self.get_images_by_id(img_id)

            if images_p:
                for image in images_p:
                    with open(save_file + image[0], 'wb') as f:
                        f.write(image[2])
                        LOG.debug('wrote image (no classify): %s' % image[0])
                time.sleep(0.5)
            else:
                LOG.debug('Ignored id: %s' % img_id)
        LOG.info('End for write image in %s' % save_file)

    def craw_rank(self, id_set=None, page=4, mode='daily', date=''):
        if not id_set:
            id_set = set()
        if not date:
            date = self.times

        # check page
        if page < 1 or page > 10:
            LOG.error("Need page>0 and <10")
            raise ValueError

        # check daily ranking
        id_set = self.scan_ranking(mode=mode, date=date, id_set=id_set, page=page)
        file_name = self.out_path + '/Daily_Rank_' + date + '/'
        self.crawler(id_set=id_set, save_file=file_name)

        LOG.info('craw_rank Final')

    def craw_artist(self, uid, mode='works', id_set=None, classify=False):
        if not id_set:
            id_set = set()
        # check works/bookmarks of artist
        id_set = self.scan_artist(uid=uid, class_=mode, id_set=id_set)
        file_name = self.out_path + '/' + uid + '_' + mode + '_' + str(len(id_set)) + 'p/'
        self.crawler(id_set=id_set, save_file=file_name)

        LOG.info('Crawler Final')


###########################
# RUN
"""
opts, args = getopt.getopt(sys.argv[1:], 'p:u:m:d:c:', ['no-classify', 'mode=', 'date=', 'uid=', 'class=', 'page='])

for name, value in opts:
    print(name, value)

    if name in ('-d', '--date'):
        DATE = value

    if name in ('-m', '--mode'):
        if value in ('rank', 'artist'):
            MODE = value
        else:
            print('Bad value of --mode')
            raise ValueError

    if name in ('-u', '--uid'):
        UID = value

    if name in ('-c', '--class'):
        if value in ('works', 'bookmarks', 'daily', 'weekly', 'monthly'):
            CLASS = value
        else:
            print('Bad value of --class')
            raise ValueError

    if name in ('-p', '--page'):
        PAGE = int(value)

for ar in args:
    if ar == "no-classify":
        classify = False

if 'classify' not in dir():
    classify = True

if 'MODE' not in dir():
    MODE = 'rank'

if MODE == 'rank':
    if 'CLASS' not in dir():
        CLASS = 'daily'
    if 'PAGE' not in dir():
        PAGE = 4
elif MODE == 'artist':
    if 'CLASS' not in dir():
        CLASS = 'works'

print(classify)
c = Crawler(user=ID, pw=PW, out_path=SAVE_PATH, date=DATE)

if MODE == 'rank':
    c.craw_rank(page=PAGE, classify=classify, mode=CLASS)
elif MODE == 'artist':
    c.craw_artist(uid=UID, classify=classify, mode=CLASS)

if conf.CREATE_DEMO:
    LOG.info('Create page')
    dc = DemoCreator(image_path='Daily_Rank_' + DATE + '/', date=DATE)
    dc.create_rank_page()
    dc.update_index()

    LOG.info('Start reduce image')
    resize_img(DATE, '/po/')
    LOG.info('END reduce image (po)')
    resize_img(DATE, '/ne/')
    LOG.info('END reduce image (ne)')

    LOG.info('Mission complete')
"""
