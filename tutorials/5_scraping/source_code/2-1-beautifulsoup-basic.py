#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网络爬虫

    安装Beautiful： bs4
    需要安装解析库模块： lxml, html5lib，xml

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen

# if has Chinese, apply decode()
html = urlopen("https://morvanzhou.github.io/static/5_scraping/basic-structure.html").read().decode('utf-8')

soup = BeautifulSoup(html, features='lxml')
print(soup.h1)
print('\n', soup.p)

all_href = soup.find_all('a')
all_href = [l['href'] for l in all_href]
print('\n', all_href)



