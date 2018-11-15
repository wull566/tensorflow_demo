#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
循环链接爬虫读取
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random


base_url = "https://baike.baidu.com"
his = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]

for i in range(20):
    # dealing with Chinese symbols
    url = base_url + his[-1]

    html = urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html, features='lxml')
    print(i, soup.find('h1').get_text(), '    url: ', his[-1])

    # find valid urls
    sub_urls = soup.find_all("a", {"target": "_blank", "href": re.compile("/item/(%.{2})+$")})

    if len(sub_urls) != 0:
        # [<a href="/item/%E9%95%9C%E5%83%8F" target="_blank">镜像</a>]
        his.append(random.sample(sub_urls, 1)[0]['href'])
    else:
        # no valid sub link found
        # 未找到返回上一层
        his.pop()
