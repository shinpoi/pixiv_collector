# - coding: utf-8 -*-

"""
daily task
"""

import getopt
import sys
import conf
import time
import datetime
import logging

from crawler.crawler import Crawler
from demo_creator import DemoCreator, resize_img

LOG = logging.getLogger('logger.main')

opts, args = getopt.getopt(sys.argv[1:], 'p:u:m:d:c:', ['no-classify', 'mode=', 'date=', 'uid=', 'class=', 'page='])


# DATE = Newest day of ranking.
today = datetime.date.today()
if int(time.strftime('%H')) < 12:
    oneday = datetime.timedelta(days=2)
else:
    oneday = datetime.timedelta(days=1)
yesterday = today - oneday

DATE = yesterday.strftime('%Y%m%d')


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

c = Crawler(user=conf.P_ID, pw=conf.P_PW, out_path=conf.CRAWLER_OUT_DIR, date=DATE)

if MODE == 'rank':
    c.craw_rank(page=PAGE, mode=CLASS)
elif MODE == 'artist':
    c.craw_artist(uid=UID, mode=CLASS)

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

    LOG.info('ALL finished')
