import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

stdsc = StandardScaler()
data = pd.read_csv('gold_data.csv')
x_dataframe = pd.read_csv('gold_data.csv', usecols=['settle'])#获取作为输入的数据
y_dataframe = pd.read_csv('gold_data.csv', usecols=['settle'])#获取作为输出的数据

x_origin = x_dataframe.as_matrix(columns=None)#输入矩阵
y_origin = y_dataframe.as_matrix(columns=None)#输出矩阵

y_ad = y_origin[30:2476]#去除前30天


# y_ad = []#生成与前5日数据对应的当日输出数据
# for i in range(len(y_origin)-5):
#     y_ad.append(y_origin[i+5])
num_data = x_origin.shape[0]
#30日调整后的输入X
x_ad = np.zeros(shape=(1,30))
for i in range(num_data-30):
    #x_ad = x_origin[0:30]
    x_temp = x_origin[i+1:i+31]
    x_temp_single = x_temp.reshape(1, 30)
    x_ad = np.append(x_ad, x_temp_single, axis=0)
x_ad = np.delete(x_ad,0,axis=0)


x = stdsc.fit_transform(x_ad)
y = stdsc.fit_transform(y_ad)# 数据归一化
x_train = x[0:1800]
x_test = x[1800:2446]
y_train = y[0:1800]
y_test = y[1800:2446]
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)#分割数据集

x1 = range(1,len(y_train)+1)
x2 = range(len(y_train)+1,len(y_train)+len(y_test)+1)
plt.plot(x1,y_train,'b')
plt.plot(x2,y_test,'r')
plt.show()