import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm
import matplotlib
matplotlib.use("TkAgg")
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import matplotlib.pyplot as plt
from ELM import HiddenLayer

data = pd.read_csv('gold_data.csv',usecols=['settle'])
x = data['settle']
y = x
x = x.as_matrix(columns=None)#输入矩阵
y = y.as_matrix(columns=None)
X = x#ELM专用



decomposer = EMD(x)
imfs = decomposer.decompose()

p_days = 1#预测p_days后的结果
p = 6#用前p天的数据预测
C = 10**8#正则化因子
t = 300#时间分割点





for i in range(6):#分解
    x_imfs = imfs[i]
    y_imfs = x#分解后的y
    num_data = x.shape[0]
    y_in = y_imfs[p + p_days - 1:num_data]#去除y的前p个数据
    x_in = np.zeros(shape=(1,p))#生成一列p行的输入矩阵
    for j in range(num_data-p+1-p_days):#将原始x按照每p个一组转为矩阵  -p_days是因为最后一天要预测故前置
        x_temp = x_imfs[j:j+p]#截取p个数为矩阵的一列
        x_temp_single = x_temp.reshape(1, p)
        x_in = np.append(x_in, x_temp_single, axis=0)#将生成的列插入矩阵后
    x_in = np.delete(x_in, 0, axis=0)#删除第一行的0
    y_in.reshape(-1,1)


    num_xt = x_in.shape[0] - p
    num_yt = y_in.shape[0] - p
    num_x = x_in.shape[0]
    num_y = y_in.shape[0]
    Y_test = y[t:num_yt]#未分解的Y_test

    stdsc = StandardScaler()
    x_in = stdsc.fit_transform(x_in)
    y_in = stdsc.fit_transform(y_in.reshape(-1,1)) # 数据归一化



    #分割数据集

    x_train = x_in[0:t]
    x_test = x_in[t:num_xt]
    y_train = y_in[0:t]
    y_test = y_in[t:num_yt]#分解的y_test
    x_validation = x_in[num_xt:num_x]
    y_validation = y_in[num_yt:num_y]


    my_ELM = HiddenLayer(x_train,20,C)
    beta = my_ELM.regressor_train(y_train)

    y_pred = my_ELM.regressor_test(x_test)
    y_pred_full = my_ELM.regressor_test(x_in)
    y_pred = stdsc.inverse_transform(y_pred)#逆归一化
    y_test = stdsc.inverse_transform(y_test)#逆归一化
    #y_pred_full = my_ELM.regressor_test(y_pred_full)#逆归一化

    MSE_0 = round(sm.mean_squared_error(Y_test,y_pred), 5)
    num_x_test = x_test.shape[0] + 1
    x_plot = list(range(1, num_x_test))

    if i ==0:
        y_pred_1 = y_pred
        y_test_1 = y_test
    elif i ==1:
        y_pred_2 = y_pred
        y_test_2 = y_test
    elif i ==2:
        y_pred_3 = y_pred
        y_test_3 = y_test
    elif i ==3:
        y_pred_4 = y_pred
        y_test_4 = y_test
    elif i ==4:
        y_pred_5 = y_pred
        y_test_5= y_test
    else:
        y_pred_6 = y_pred
        y_test_6 = y_test


#全数据
# y_in_full = y[p + p_days - 1:num_data]#去除y的前p个数据
# x_in_full = np.zeros(shape=(1,p))#生成一列p行的输入矩阵
# for k in range(num_data-p+1-p_days):#将原始x按照每p个一组转为矩阵  -p_days是因为最后一天要预测故前置
#     x_temp_full = x_in_full[j:j+p]#截取p个数为矩阵的一列
#     x_temp_single_full = x_temp.reshape(1, p)
#     x_in = np.append(x_in, x_temp_single_full, axis=0)#将生成的列插入矩阵后
# x_in_full = np.delete(x_in, 0, axis=0)#删除第一行的0
# y_in_full.reshape(-1,1)





Y_pred = y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4 + y_pred_5+ y_pred_6
MSE_1 = round(sm.mean_squared_error(y_test_1,y_pred_1), 5)
MSE_2 = round(sm.mean_squared_error(y_test_2,y_pred_2), 5)
MSE_3 = round(sm.mean_squared_error(y_test_3,y_pred_3), 5)
MSE_4 = round(sm.mean_squared_error(y_test_4,y_pred_4), 5)
MSE_5 = round(sm.mean_squared_error(y_test_5,y_pred_5), 5)
MSE_6 = round(sm.mean_squared_error(y_test_6,y_pred_6), 5)
num_x_test = x_test.shape[0] + 1
x_plot = list(range(1, num_x_test))
print('mean squared error1=', MSE_1)  # 均方误差
print('mean squared error2=', MSE_2)  # 均方误差
print('mean squared error3=', MSE_3)  # 均方误差
print('mean squared error4=', MSE_4)  # 均方误差
print('mean squared error5=', MSE_5)  # 均方误差
print('mean squared error6=', MSE_6)  # 均方误差

fig1 = plt.figure(figsize=(12,9))
plt.subplot(6,1,1)
plt.plot(x_plot,y_pred_1, label='EMD-ELM1')
plt.plot(x_plot,y_test_1, label='TEST')
plt.legend()
plt.subplot(6,1,2)
plt.plot(x_plot,y_pred_2, label='EMD_ELM2')
plt.plot(x_plot,y_test_2, label='TEST')
plt.legend()
plt.subplot(6,1,3)
plt.plot(x_plot,y_pred_3, label='EMD_ELM3')
plt.plot(x_plot,y_test_3, label='TEST')
plt.legend()
plt.subplot(6,1,4)
plt.plot(x_plot,y_pred_4, label='EMD_ELM4')
plt.plot(x_plot,y_test_4, label='TEST')
plt.legend()
plt.subplot(6,1,5)
plt.plot(x_plot,y_pred_5, label='EMD_ELM5')
plt.plot(x_plot,y_test_5, label='TEST')
plt.legend()
plt.subplot(6,1,6)
plt.plot(x_plot,y_pred_6, label='EMD_ELM6')
plt.plot(x_plot,y_test_6, label='TEST')
plt.legend()
plt.show()



# print(y_pred_1.shape)
# print(y_pred_2.shape)
# print(y_pred_3.shape)
# print(y_pred_4.shape)
# print(y_pred_5.shape)
# print(y_pred_6.shape)
# print(Y_pred)
# print(Y_pred.shape)

MSE = round(sm.mean_squared_error(Y_test, Y_pred), 5)
num_x_test = x_test.shape[0]+1
x_plot = list(range(1,num_x_test))
print('mean squared error EMD_ELM=',MSE)#均方误差


#单一ELM对比
num_data_0 = X.shape[0]
y_in_0 = y[p + p_days - 1:num_data_0]#去除y的前p个数据
x_in_0 = np.zeros(shape=(1,p))#生成一列p行的输入矩阵
for k in range(num_data_0-p-p_days+1):#将原始x按照每p个一组转为矩阵  -p_days是因为最后一天要预测故前置
    x_temp_0 = X[k:k+p]#截取p个数为矩阵的一列
    x_temp_single_0 = x_temp_0.reshape(1, p)
    x_in_0 = np.append(x_in_0, x_temp_single_0, axis=0)#将生成的列插入矩阵后
x_in_0 = np.delete(x_in_0, 0, axis=0)#删除第一行的0
y_in_0.reshape(-1,1)


num_xt_0 = X.shape[0] - p
num_yt_0 = y.shape[0] - p
num_x_0 = X.shape[0]
num_y_0 = y.shape[0]


stdsc = StandardScaler()
x_0 = stdsc.fit_transform(x_in_0)
y_0 = stdsc.fit_transform(y_in_0.reshape(-1,1)) # 数据归一化



#分割原始数据集

x_train_0 = x_0[0:t]
x_test_0 = x_0[t:num_xt_0]
y_train_0 = y_0[0:t]
y_test_0 = y_0[t:num_yt_0]
x_validation_0 = x_0[num_xt_0:num_x_0]
y_validation_0 = y_0[num_yt_0:num_y_0]

#单一ELM
my_ELM_0 = HiddenLayer(x_train_0,20,C)
beta_0 = my_ELM_0.regressor_train(y_train_0)
y_pred_0 = my_ELM.regressor_test(x_test_0)
y_pred_0 = stdsc.inverse_transform(y_pred_0)


#全数据预测
y_pred_0_full = my_ELM_0.regressor_test(x_0)
y_pred_0_full = stdsc.inverse_transform(y_pred_0_full)


MSE_ELM = round(sm.mean_squared_error(Y_test, y_pred_0), 5)
print('mean squared error ELM=',MSE_ELM)#均方误差


fig2 = plt.figure(figsize=(12,6))
plt.plot(x_plot,Y_pred,c='b',label='EMD-ELM')
plt.plot(x_plot,Y_test,c='r',label='TEST')
plt.plot(x_plot,y_pred_0,c='y',label='ELM')
plt.legend()
plt.show()

fig3 = plt.figure(figsize =(12,6))
plt.plot(y_pred_full,c='b',label='EMD-ELM')
plt.plot(y,c='r',label='TEST')
plt.plot(y_pred_0_full,c='y',label='ELM')
plt.legend()
plt.show()





