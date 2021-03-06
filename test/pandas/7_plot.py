# View more 3_python 2_tensorflow_old on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for 3_python 3+. If you are using 3_python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plot data

# Series
# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
# data = data.cumsum()
# data.plot()

# DataFrame
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
# data.plot()

# plot methods 图像类型:
# 'bar', 'hist', 'box', 'kde', 'area', scatter', hexbin', 'pie'
# scatter 打印点
# ax 代表在同一个图片上
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label="Class 1")
data.plot.scatter(x='A', y='C', color='LightGreen', label='Class 2', ax=ax)

plt.show()