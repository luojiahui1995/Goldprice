import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

#载入时间序列数据
data = pd.read_csv('gold_data_.csv',usecols=['settle'])
data = data['settle']
decomposer = EMD(data)
imfs = decomposer.decompose()
print(imfs)
#绘制分解图
plot_imfs(data,imfs,data.index)
#保存IMFs
arr = np.vstack((imfs,data))
dataframe = pd.DataFrame(arr.T)
dataframe.to_csv('imf.csv',index=None,columns=None)
print(imfs[0])