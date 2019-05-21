import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

#载入时间序列数据
data = pd.read_csv('gold_data.csv',usecols=['settle'])
#EMD经验模态分解
decomposer = EMD(data[0])
imfs = decomposer.decompose()
#绘制分解图
plot_imfs(data[0],imfs,data.index)
#保存IMFs
arr = np.vstack((imfs,data[0]))
dataframe = pd.DataFrame(arr.T)
dataframe.to_csv('C:\imf.csv',index=None,columns=None)
