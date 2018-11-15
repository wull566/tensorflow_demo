# View more 3_python 2_tensorflow_old on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
"""
Please note, this code is only for 3_python 3+. If you are using 3_python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import pandas as pd

# read from
data = pd.read_csv('student.csv')
print(data)

# save to
data.to_pickle('student.pickle')