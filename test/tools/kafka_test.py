#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kafka 练习

"""

from __future__ import print_function
from pykafka import KafkaClient

# 192.168.0.145:27017
client = KafkaClient(hosts = "172.16.82.163:9091")