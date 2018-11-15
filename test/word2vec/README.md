# Word2vec 词向量学习

## 安装 gensim 文档语义库， 内含 word2vec python 调用库

    pip3 install gensim


## 

    安装 Microsoft Visual C++ Build Tools 14.0 以上版本
    
    https://visualstudio.microsoft.com/zh-hans/downloads/
    下载 Visual Studio 2017 生成工具
    
    https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15

    pip3 install Cython
    pip3 install word2vec


## 安装测试

#### 英文编译测试

(1)到官网到下载：https://code.google.com/archive/p/word2vec/，然后选择export 到github，也可以直接到我的github克隆：

    git clone https://github.com/hjimce/word2vec.git

(2)编译：make

(3)下载测试数据http://mattmahoney.net/dc/text8.zip，并解压

(4)输入命令train起来：

    time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15

(5)测试距离功能：

    ./distance vectors.bin


#### 中文训练测试

(1)中文词向量：下载数据msr_training.utf8，这个数据已经做好分词工作，如果想要直接使用自己的数据，就需要先做好分词工作

(2)输入命令train起来： 

    time ./word2vec -train msr_training.utf8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15

(3)启动相似度距离测试：

    ./distance vectors.bin

(4)输入相关中文词：中国，查看结果：