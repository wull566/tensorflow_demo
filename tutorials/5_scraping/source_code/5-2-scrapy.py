#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

爬虫高级框架 scrapy 商业化

https://github.com/scrapy/scrapy

中文官网:
https://scrapy-chs.readthedocs.io/zh_CN/0.24/

Spiders爬虫模块， downloder下载模块， item pipelines

注意：安装scrapy模块，window可能报错
    error: Microsoft Visual C++ 14.0 is required
    需要安装对应版本Twisted，需要下载包后安装 https://www.lfd.uci.edu/~gohlke/pythonlibs/#twisted
        pip3 install Twisted-18.7.0-cp36-cp36m-win_amd64.whl
    重新安装scrapy模块

"""

import scrapy


class MofanSpider(scrapy.Spider):
    name = "mofan"
    start_urls = [
        'https://morvanzhou.github.io/',
    ]
    # unseen = set()
    # seen = set()      # we don't need these two as scrapy will deal with them automatically

    def parse(self, response):
        # yield 体现异步处理
        yield {     # return some results
            'title': response.css('h1::text').extract_first(default='Missing').strip().replace('"', ""),
            'url': response.url,
        }

        urls = response.css('a::attr(href)').re(r'^/.+?/$')     # find all sub urls
        for url in urls:
            yield response.follow(url, callback=self.parse)     # it will filter duplication automatically


# lastly, run this in terminal
# scrapy runspider 5-2-scrapy.py -o res.json