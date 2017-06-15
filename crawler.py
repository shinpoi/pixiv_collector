# -*- coding: utf-8 -*-
# python 3.7

import os
import sys
import time
import datetime
import logging
import re
import getopt
import requests
from bs4 import BeautifulSoup
import json
import cv2
import numpy as np
# local file
import setting
import predictor
from demo_creator import DemoCreator, reduce_img

# If used Python2.7, de-command next two col. // Set encode: utf-8
# reload(sys)
# sys.setdefaultencoding('utf8')

# Parameter
ID = setting.PIXIV_ID
PW = setting.PIXIV_PW

LOG_DIR = sys.path[0] + '/' + setting.LOG_DIR
SAVE_PATH = sys.path[0] + '/' + setting.CRAWLER_DIR
LOG_LEVEL = logging.INFO
LOG_TO_CONSOLE = True


for d in [LOG_DIR, SAVE_PATH]:
    try:
        os.mkdir(d)
    except OSError:
        pass


# DATE = Newest day of ranking.
today = datetime.date.today()
if int(time.strftime('%H')) < 12:
    oneday = datetime.timedelta(days=2)
else:
    oneday = datetime.timedelta(days=1)
yesterday = today-oneday

DATE = yesterday.strftime('%Y%m%d')


# Set logging
LOG_NAME = 'Crawler_' + time.strftime('%Y-%m-%d_%H-%M') + '.log'
logging.basicConfig(level=setting.LOG_LEVEL,
                    format='[%(levelname)s]   \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    filename=LOG_DIR + LOG_NAME,
                    filemode='a'
                    )

if LOG_TO_CONSOLE:
    console = logging.StreamHandler()
    console.setLevel(setting.LOG_LEVEL)  # setting.LOG_LEVEL
    formatter = logging.Formatter('[%(levelname)s]  \t%(message)s\t')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# Pixiv Crawler
class Crawler(object):
    def __init__(self, user, pw, save_path, date):
        self.id = user
        self.pw = pw
        self.uid = ''
        self.path = save_path
        self.times = date

        self.domain = 'www.pixiv.net'
        self.homepage = 'http://www.pixiv.net/'
        self.pre_login_page = 'https://accounts.pixiv.net/login?lang=en&source=pc&view_type=page&ref=wwwtop_accounts_index'
        self.login_page = 'https://accounts.pixiv.net/api/login?lang=zh'
        self.image_page = 'http://www.pixiv.net/member_illust.php?mode=medium&illust_id='
        self.manga_page = 'http://www.pixiv.net/member_illust.php?mode=manga&illust_id='
        self.score_page = 'http://www.pixiv.net/rpc_rating.php'
        self.ranking_page = 'http://www.pixiv.net/ranking.php?'
        self.works_page = 'http://www.pixiv.net/member_illust.php?type=all&id='
        self.bookmarks_page = 'http://www.pixiv.net/bookmark.php?rest=show&id='

        self.headers_base = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': self.homepage,
        }

        self.pattern_host = re.compile('https?://(.+?)/')
        self.pattern_tt = re.compile('pixiv.context.token = "([a-zA-Z0-9]+?)";')
        self.pattern_uid = re.compile('id=([0-9]+)')
        self.pattern_img_name = re.compile('[0-9]+_p[0-9]+')
        self.img_value = {}

        self.cookies = self.load_cookies()

    @staticmethod
    # Need bit data of image. | return a bool
    def classifiter(data, f=predictor.EasyInceptionV2()):
        # return True
        data = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 0)
        return f.predict(data)[0]

    # Load cookies from file. | Success: read cookies ; False: login and get new cookies.
    def load_cookies(self):
        try:
            with open(self.path + 'cookies.json', 'r') as f:
                cookies = requests.utils.cookiejar_from_dict(json.loads(f.read()))
            logging.info('read cookies')
            cookies = self.check_cookies(cookies)

            # check dose cookies is overdue
            if cookies:
                pass
            else:
                logging.warning('Cookies is overdue!')
                cookies = self.login()

        except IOError:
            logging.warning("Didn't find cookies")
            cookies = self.login()

        return cookies

    # Save cookies into file.
    def save_cookies(self, cookies):
        cookies = requests.utils.dict_from_cookiejar(cookies)
        with open(self.path + 'cookies.json', 'w') as f:
            f.write(json.dumps(cookies))
        logging.info('cookies was saved')

    # Save data into file. (Need name includes relativepath)
    def save_file(self, name, src):
        if not src:
            logging.error("save_file(): Did't get data")
            return False
        with open(self.path + name, 'wb') as f:
            f.write(src)
        logging.info('File %s saved' % self.path + name)

    @staticmethod
    # Update cookies. | Success:  updated cookies;
    def update_cookies(old_cookies, new_cookies):
        origin = requests.utils.dict_from_cookiejar(old_cookies)
        neo = requests.utils.dict_from_cookiejar(new_cookies)
        for i in neo:
            origin[i] = neo[i]
        neo = requests.utils.cookiejar_from_dict(origin)
        logging.info('update cookies')
        return neo

    # Check cookies by access homepage. | Success: cookies ; False: False(bool)
    def check_cookies(self, cookies):
        # Get Homepage
        logging.info('access homepage')
        headers = self.headers_base
        headers['Host'] = self.domain
        time.sleep(1)

        r = requests.get(self.homepage, headers=headers, cookies=cookies)
        soup = BeautifulSoup(r.text, 'lxml')
        check_id = soup.find(text=self.id)
        if not check_id:
            logging.error('Can not get uid from homepage')
            return False
        
        # Set uid
        uid = check_id.parent['href']
        self.uid = re.search(self.pattern_uid, uid).group(1)
        new_cookies = self.update_cookies(cookies, r.cookies)
        self.save_cookies(new_cookies)
        return new_cookies

    # Login and get new cookies. | Success: new cookies ; False: raise ValueError (and break program)
    def login(self):
        # Pre-Login
        logging.info('Login Pixiv ...')
        r = requests.get(self.pre_login_page)
        logging.info('Pre-Login... status_code: %s' % r.status_code)
        cookies = r.cookies
        headers = self.headers_base
        soup = BeautifulSoup(r.text, 'lxml')

        # Login
        headers['Referer'] = self.pre_login_page
        headers['Host'] = re.search(self.pattern_host, self.login_page).group(1)

        f = soup.find("form", action="/login")
        post_data = {}
        for i in f.find_all('input'):
            try:
                post_data[i['name']] = i['value']
            except KeyError:
                post_data[i['name']] = ''
        post_data['pixiv_id'] = self.id
        post_data['password'] = self.pw

        r = requests.post(self.login_page, data=post_data, headers=headers, cookies=cookies)
        logging.info('Login... status_code: %s' % r.status_code)
        time.sleep(1)
        cookies = self.check_cookies(r.cookies)
        if cookies:
            logging.info("Login successful. ID = %s " % self.id)
        else:
            logging.error("Login Failed! ID = %s " % self.id)
            raise ValueError
        return cookies

    @staticmethod
    def image_format(url):
        pattern_img_url = re.compile('/[0-9]+_p[0-9]+.(jpg|gif|png|bmp)')
        name = re.search(pattern_img_url, url).group(0)
        img_format = re.search(pattern_img_url, url).group(1)
        return name[1:], img_format

    # Get original image by image_id. | Success(illust): ((name, format, data), )  <--tuple ;
    #                                   Success(manga): [(name 1, format 1, data 1), (name 2, format 2, data 2), ...)] <--list ;
    #                                   False: None
    def get_images_by_id(self, img_id, referer=None):
        # get url of original image
        url = self.image_page + img_id
        headers = self.headers_base
        if referer:
            headers['Referer'] = referer
        headers['Host'] = self.domain

        # access medium image page
        r = requests.get(url, headers=headers, cookies=self.cookies)
        soup = BeautifulSoup(r.text, 'lxml')

        # illust
        # get url of original image
        img_url = soup.find('img', class_='original-image')
        if img_url:
            img_url = img_url['data-src']
            # get original image
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36',
                'Referer': url
            }

            name, img_format = self.image_format(img_url)

            # .GTF files will be dealt with next Version (Maybe... #1
            if img_format == 'gif':
                return None
            img = requests.get(img_url, headers=headers)
            return (name, img_format, img.content),

        # manga
        manga = soup.find('div', class_='multiple')
        if manga:
            img_url = self.manga_page + img_id

            # access manga page
            r = requests.get(img_url, headers=headers, cookies=self.cookies)
            soup = BeautifulSoup(r.text, 'lxml')

            # get list of img
            tag_list = soup.find_all('img', class_='image')
            page = len(tag_list)

            # If a image-id has more than 10 images,  ignored it.
            if page > 10:
                logging.info('(too more images) Ignored id: %s ' % img_id)
                return None

            # get original image
            headers['Referer'] = img_url
            manga_page = "http://www.pixiv.net/member_illust.php?mode=manga_big&illust_id=" + img_id + "&page="
            img_list = []
            for i in range(page):
                # get url of original image
                url = manga_page + str(i)
                r = requests.get(url, headers=headers, cookies=self.cookies)
                soup = BeautifulSoup(r.text, 'lxml')
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36',
                    'Referer': url
                }

                url = soup.find('img')['src']
                #
                #
                # Filter!!!!
                # continue
                #

                # get original image
                name, img_format = self.image_format(url)
                # .GTF files will be dealt with next Version (Maybe... #2
                if img_format == 'gif':
                    return None
                img = requests.get(url, headers=headers)
                img_list.append((name, img_format, img.content))
                time.sleep(0.5)
            return img_list

        return None

    # Score image by image_id.
    def score(self, img_id, tt, score='10'):
        url = self.image_page + img_id
        headers = self.headers_base
        headers['Referer'] = url
        headers['Host'] = self.domain

        data = {
            'mode': 'save',
            'i_id': img_id,
            'u_id': self.uid,
            'qr': '0',
            'score': score,
            'tt': tt,
        }

        r = requests.post(self.score_page, data=data, headers=headers, cookies=self.cookies)
        try:
            res = json.loads(r.text)
            if res['re_sess']:
                logging.info('Score image(id=%s) by %s' % (img_id, score))
            else:
                logging.warning('Score image(id=%s) failed \t\n response: %s' % (img_id, r.text))
        except ValueError:
            logging.error('Score image(id=%s) failed (didn\'t get response)' % img_id)
    """
    # Add image-id of (daily) ranking into id-set-po and id-set-ne. | Success: (id-set-po, id-set-ne) <- as a tuple;
    def scan_ranking_preview(self, mode='daily', content='illust', page=4, date=''):
        if date:
            pass
        else:
            date = self.times

        id_set_po = set()
        id_set_ne = set()
        logging.info('Start scan ranking (predict mode).(mode=%s, content=%s, date=%s)' % (mode, content, date))

        overview_url = self.ranking_page + 'mode=' + mode + '&content=' + content + '&date=' + date
        headers = self.headers_base
        headers['Referer'] = self.homepage
        headers['Host'] = self.domain
        r = requests.get(overview_url, headers=headers, cookies=self.cookies)
        tt = re.search(self.pattern_tt, r.text).group(1)
        headers['Referer'] = overview_url

        for i in range(page):
            headers['Host'] = self.domain
            para = 'mode=' + mode + '&content=' + content + '&date=' + date + '&p=' + str(i + 1) + '&format=json&tt=' + tt
            json_url = self.ranking_page + para
            json_rank = json.loads(requests.get(json_url, headers=headers, cookies=self.cookies).text)
            headers['Host'] = re.match(self.pattern_host, json_rank['contents'][0]['url']).group(1)
            logging.info("scanning page %d" % i)
            try:
                for img_json in json_rank['contents']:
                    url = img_json['url']
                    img = requests.get(url, headers=headers, cookies=self.cookies)
                    flag = self.classifiter(img.content)
                    if flag:
                        id_set_po.add(str(img_json['illust_id']))
                        logging.debug('add %s into positive-set' % img_json['illust_id'])
                    else:
                        id_set_ne.add(str(img_json['illust_id']))
                        logging.debug('add %s into negative-set' % img_json['illust_id'])
                    time.sleep(0.3)
            except KeyError:
                print(json_rank)
                logging.error('Get json of page %s failed' % str(i + 1))

        logging.info('End scan ranking (predict mode).')
        return id_set_po, id_set_ne
    """
    # Add image-id of (daily) ranking into id-set. | Success: id-set ;
    def scan_ranking(self, mode='daily', content='illust', page=4, date='', id_set=set()):
        if date:
            pass
        else:
            date = self.times

        logging.info('Start scan ranking.(mode=%s, content=%s, date=%s)' % (mode, content, date))

        overview_url = self.ranking_page + 'mode=' + mode + '&content=' + content + '&date=' + date
        headers = self.headers_base
        headers['Referer'] = self.homepage
        headers['Host'] = self.domain
        r = requests.get(overview_url, headers=headers, cookies=self.cookies)
        tt = re.search(self.pattern_tt, r.text).group(1)
        headers['Referer'] = overview_url

        for i in range(page):
            para = 'mode=' + mode + '&content=' + content + '&date=' + date + '&p=' + str(i+1) + '&format=json&tt=' + tt
            json_url = self.ranking_page + para
            json_rank = json.loads(requests.get(json_url, headers=headers, cookies=self.cookies).text)
            try:
                for img_json in json_rank['contents']:
                    id_set.add(str(img_json['illust_id']))
                time.sleep(0.5)
            except KeyError:
                print(json_rank)
                logging.error('Get json of page %s failed' % str(i+1))

        logging.info('End scan ranking.')
        return id_set

    # Add artist's works/bookmarks image-id into id-set. | Success: id-set ;
    def scan_artist(self, uid, class_='works', id_set=set()):
        logging.info('Start scan user(id=%s), scan_type is: %s.' % (uid, class_))

        if class_ == 'works':
            page = self.works_page
        elif class_ == 'bookmarks':
            page = self.bookmarks_page
        else:
            logging.error('scan_artist(): Get a wrong value of class_: %s' % class_)
            raise ValueError

        headers = self.headers_base
        headers['Host'] = self.domain
        max_page = '0'
        new_max_page = '1'

        while max_page != new_max_page:
            max_page = new_max_page
            url = page + uid + '&p=' + max_page
            r = requests.get(url, headers=headers, cookies=self.cookies)
            soup = BeautifulSoup(r.text, 'lxml')
            new_max_page = soup.find('ul', class_='page-list')
            if new_max_page:
                new_max_page = new_max_page.find_all('li')[-1].text
            else:
                new_max_page = '1'
                break

        logging.info('Has %s pages!' % new_max_page)
        for i in range(int(max_page)):
            url = page + uid + '&p=' + str(i+1)
            r = requests.get(url, headers=headers, cookies=self.cookies)
            soup = BeautifulSoup(r.text, 'lxml')
            tag_list = soup.find_all('img', class_='ui-scroll-view')
            for tag in tag_list:
                id_set.add(tag['data-id'])
            time.sleep(0.5)

        logging.info('End scan user')
        return id_set

    # Save original image by id-set.
    def crawler(self, id_set, save_file, classify=True):
        try:
            os.mkdir(save_file)
            os.mkdir(save_file + 'po/')
            os.mkdir(save_file + 'ne/')
        except OSError:
            pass

        logging.info('Start for write image in %s' % save_file)
        num = str(len(id_set))
        n = 0
        for img_id in id_set:
            n += 1
            logging.info("check image %s/%s" % (str(n), num))
            images_p = self.get_images_by_id(img_id)

            if images_p:
                # classify = True
                if classify:
                    for image in images_p:
                        res, rating = self.classifiter(image[2])
                        img_name = self.pattern_img_name.search(image[0])
                        self.img_value[img_name] = '[%f, %f]' % (rating[0], rating[1])
                        if res:
                            with open(save_file + 'po/' + image[0], 'wb') as f:
                                f.write(image[2])
                                logging.debug('wrote image (positive): %s' % image[0])
                        else:
                            with open(save_file + 'ne/' + image[0], 'wb') as f:
                                f.write(image[2])
                            logging.debug('wrote image (negative): %s' % image[0])
                    time.sleep(0.5)
                # classify = False
                else:
                    for image in images_p:
                        with open(save_file + image[0], 'wb') as f:
                            f.write(image[2])
                            logging.debug('wrote image (no classify): %s' % image[0])
                    time.sleep(0.5)
            else:
                logging.info('Ignored id: %s' % img_id)
        with open(save_file + 'value.json', 'w') as f:
            json.dump(self.img_value, f)
            logging.info('rating of images has saved as %s', save_file + 'value.json')
        logging.info('End for write image in %s' % save_file)

    def craw_rank(self, id_set=set(), page=4, mode='daily', date='', classify=True):
        # check page
        if page < 1 or page > 10:
            print("Need page>0 and <10")
            raise ValueError

        if date:
            day = date
        else:
            day = self.times

        # check daily ranking
        id_set = self.scan_ranking(mode=mode, date=day, id_set=id_set, page=page)
        file_name = self.path + 'Daily_Rank_' + day + '/'
        self.crawler(id_set=id_set, save_file=file_name, classify=classify)

        self.check_cookies(self.cookies)
        logging.info('Crawler Final')

    def craw_artist(self, uid, mode='works', id_set=set(), classify=False):
        # check works/bookmarks of artist
        id_set = self.scan_artist(uid=uid, class_=mode, id_set=id_set)
        file_name = self.path + uid + '_' + mode + '_' + str(len(id_set)) + 'p/'
        self.crawler(id_set=id_set, save_file=file_name, classify=classify)

        self.check_cookies(self.cookies)
        logging.info('Crawler Final')


###########################
# RUN

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
        PAGE = value

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
c = Crawler(user=ID, pw=PW, save_path=SAVE_PATH, date=DATE)

if MODE == 'rank':
    c.craw_rank(page=PAGE, classify=classify, mode=CLASS)
elif MODE == 'artist':
    c.craw_artist(uid=UID, classify=classify, mode=CLASS)
    
    
if setting.CREATE_DEMO:
    logging.info('Create page')
    dc = DemoCreator(root=setting.ROOT, image_path='Daily_Rank_'+DATE+'/', date=DATE)
    dc.creat_rank_page()
    dc.update_index()
    
    logging.info('Start reduce image')
    reduce_img(DATE, '/po/')
    logging.info('END reduce image (po)')
    reduce_img(DATE, '/ne/')
    logging.info('END reduce image (ne)')
    
    logging.info('Mission complete')
