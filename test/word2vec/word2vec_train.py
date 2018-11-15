#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print(globals()['__doc__'] % locals())
    #     sys.exit(1)
    # inp, outp1, outp2 = sys.argv[1:4]
    inp, outp1, outp2 = 'data/text8', 'work/text8.model', 'work/text8.vector'


    model = Word2Vec(LineSentence(inp), size=800, window=10, min_count=5, sg=1, hs=1,
                     workers=multiprocessing.cpu_count())
    # size: 输出向量维度
    # window： 为训练的窗口大小，8表示每个词考虑前8个词与后8个词（实际代码中还有一个随机选窗口的过程，窗口大小<=5)
    # window:skip-gram通常在10附近，CBOW通常在5附近
    # hs: 如果为1则会采用hierarchica softmax技巧。如果设置为0（defaut），则negative sampling会被使用。
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)