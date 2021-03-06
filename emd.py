import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

#载入时间序列数据
data = pd.read_csv('gold_data.csv',usecols=['settle'])

#EMD经验模态分解
x = data['settle']
decomposer = EMD(x)
imfs = decomposer.decompose()

#绘制分解图
plot_imfs(x,imfs,data.index)
#保存IMFs
arr = np.vstack((imfs,x))
dataframe = pd.DataFrame(arr.T)
dataframe.to_csv('D:/imf.csv',index=None,columns=None)

