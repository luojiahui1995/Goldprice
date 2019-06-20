import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from ELM import HiddenLayer

# 数据预处理
stdsc = StandardScaler()
data = pd.read_csv('gold_data_.csv')
x_dataframe = pd.read_csv('gold_data_.csv', usecols=['settle'])#获取作为输入的数据
y_dataframe = pd.read_csv('gold_data_.csv', usecols=['settle'])#获取作为输出的数据

x_origin = x_dataframe.as_matrix(columns=None)#输入矩阵
y_origin = y_dataframe.as_matrix(columns=None)#输出矩阵

p_days = 1
num_pastdays = 30
num_data = x_origin.shape[0]
y_ad = y_origin[num_pastdays+p_days-1:num_data]#去除前30天


# y_ad = []#生成与前5日数据对应的当日输出数据
# for i in range(len(y_origin)-5):
#     y_ad.append(y_origin[i+5])


#30日调整后的输入X
x_ad = np.zeros(shape=(1,num_pastdays))
for i in range(num_data-num_pastdays+1-p_days):
    #x_ad = x_origin[0:30]
    x_temp = x_origin[i:i+num_pastdays]
    x_temp_single = x_temp.reshape(1, num_pastdays)
    x_ad = np.append(x_ad, x_temp_single, axis=0)
x_ad = np.delete(x_ad, 0, axis=0)

x = x_ad
y = y_ad
# x = stdsc.fit_transform(x_ad)
# y = stdsc.fit_transform(y_ad)# 数据归一化
num_xt = x.shape[0]-14
num_yt = y.shape[0]-14
num_x = x.shape[0]
num_y = y.shape[0]
x_train = x[0:2000]
x_test = x[2000:num_xt]
y_train = y[0:2000]
y_test = y[2000:num_yt]
x_validation = x[num_xt:num_x]
y_validation = y[num_yt:num_y]
print(x_test.shape)
print(num_xt)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)#分割数据集

# listx = []
# listy = []
# listY = []
#for i in range(0,100):
#listx.append(i)
#print(len(y_test))    #-----152个测试集样本点

#生成测试正则化因子
# list_C = []
# temp_x = []
# list_c = []
# for i in range(-4,11):
#     temp = pow(10, i)
#     temp_x.append(i)#画图横轴
#     list_C.append(temp)
# print(temp_x)

C = 10**10
#for C in list_C:
def rc_ELM(x_train, y_train,x_test,C):
    my_ELM = HiddenLayer(x_train,30, C)
    beta1 = my_ELM.regressor_train(y_train)
    y_out = my_ELM.regressor_test(x_train)
    e1 = y_train - y_out
    num_x_test = x_test.shape[0]+1
    #x = list(range(1,num_x_test))#生成与测试集样本数相同的序列用来作为X轴画图
    # x_ = []
    # for i in range(646):
    #     x_.append([x[i]])
    # x_plot = np.linspace(0.9, 5.02, 100).reshape(-1, 1)
    # y_plot = my_ELM.regressor_test(x_plot)
    y1_pred = my_ELM.regressor_test(x_test)#y~有2475个值-----------------------------预测值

    # plt.scatter(x_test, y_test, c = 'b')
    # plt.scatter(x_test, y_pred, c = 'r')
    # plt.title('ELM_regress')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    #MSE1 = round(sm.mean_squared_error(y_test, y1_pred), 5)
    #listy.append(MSE1)
   # print('mean squared error1=',MSE1)#原始ELM均方误差
    E1_ELM = HiddenLayer(x_train, 30,C)
    beta2 = E1_ELM.regressor_train(e1)
    e1_out = E1_ELM.regressor_test(x_train)
    e2 = e1 - e1_out
    e1_pred = E1_ELM.regressor_test(x_test)
    y2_pred = y1_pred + e1_pred
    #plt.plot(x, Y_pred, c='g')
    #print(e1_pred)
   # MSE2 = round(sm.mean_squared_error(y_test,y2_pred), 5)
   # print('mean squared error2=',MSE2)#一层残差层均方误差


    E2_ELM = HiddenLayer(x_train, 30, C)
    beta3 = E2_ELM.regressor_train(e2)
    e2_out = E2_ELM.regressor_test(x_train)
    e3 = e2-e2_out
    e2_pred = E2_ELM.regressor_test(x_test)
    c_1 = (1/pow(np.linalg.norm(e2),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2)))
    c_2 = (1/pow(np.linalg.norm(e3),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2)))
    y3_pred = y1_pred + c_1*e1_pred + c_2*e2_pred
    #MSE3 = round(sm.mean_squared_error(y_test,y3_pred),5)
    #print('mean squared error3=',MSE3)#二层残差层均方误差


    E3_ELM = HiddenLayer(x_train, 30, C)
    beta4 = E3_ELM.regressor_train(e3)
    e3_out = E3_ELM.regressor_test(x_train)
    e4 = e3-e3_out
    e3_pred = E3_ELM.regressor_test(x_test)
    c_1 = (1/pow(np.linalg.norm(e2),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2)))
    c_2 = (1/pow(np.linalg.norm(e3),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2)))
    c_3 = (1/pow(np.linalg.norm(e4),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2)))
    y4_pred = y1_pred + c_1*e1_pred + c_2*e2_pred + c_3*e3_pred
    #MSE4 = round(sm.mean_squared_error(y_test,y4_pred),5)
    #print('mean squared error4=',MSE4)#三层残差层均方误差


    E4_ELM = HiddenLayer(x_train, 30, C)
    beta5 = E4_ELM.regressor_train(e4)
    e4_out = E4_ELM.regressor_test(x_train)
    e5 = e4-e4_out
    e4_pred = E4_ELM.regressor_test(x_test)
    c_1 = (1/pow(np.linalg.norm(e2),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2))+(1/pow(np.linalg.norm(e5),2)))
    c_2 = (1/pow(np.linalg.norm(e3),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2))+(1/pow(np.linalg.norm(e5),2)))
    c_3 = (1/pow(np.linalg.norm(e4),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2))+(1/pow(np.linalg.norm(e5),2)))
    c_4 = (1/pow(np.linalg.norm(e5),2))/((1/pow(np.linalg.norm(e2),2))+(1/pow(np.linalg.norm(e3),2))+(1/pow(np.linalg.norm(e4),2))+(1/pow(np.linalg.norm(e5),2)))
    y5_pred = y1_pred + c_1*e1_pred + c_2*e2_pred + c_3*e3_pred + c_4*e4_pred
   #MSE5 = round(sm.mean_squared_error(y_test,y5_pred),5)
    #print('mean squared error5=',MSE5)#三层残差层均方误差

    return y3_pred


    #list_c.append(MSE1)
    # print(list_c)
    # plt.xlim(-4, 11 )
    # plt.ylim(0,100)
    # plt.plot(temp_x, list_c, marker='o', color='blue', linewidth=1.0, linestyle='--', label='linear line')
    # plt.xlabel('regularization factor log(C)')
    # plt.ylabel('MSE')
    # my_x_ticks = np.arange(-4, 11,1)
    # my_y_ticks = np.arange(0, 200, 10)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    # plt.show()


    #listY.append(MSE2)
    # plt.plot(listx, listy,'b')
    # plt.plot(listx,listY,'r')
    list_out =list(zip(y_test,y1_pred, y2_pred,y3_pred,y4_pred,y5_pred))
    # def takefirst(elem):
    #     return elem[0]
    # list_out.sort(key=takefirst)#以第一列为关键字排序
    # print(list_out)
    print(num_data)
    print(x_test.shape)
    list1 = np.array([x[0] for x in list_out])
    list2 = np.array([x[1] for x in list_out])
    list3 = np.array([x[2] for x in list_out])
    list4 = np.array([x[3] for x in list_out])
    list5 = np.array([x[4] for x in list_out])
    list6 = np.array([x[5] for x in list_out])
    print(y_test.tolist())
    #plt.subplot(2,1,1)
    plt.plot(x, list1, c='b',label='TEST')
    plt.plot(x, list2, c='r',label='ELM')
    plt.plot(x, list3, c='y',label='RC1_ELM')
    plt.plot(x, list4, c='pink',label='RC2_ELM')
    plt.ylabel('predicted value')
    plt.legend()
    #plt.scatter(x, list4, c='y')
    #plt.scatter(x, list5, c='pink')

    #plt.show()

    # list_layer = [1,2,3,4,5]
    # list_layer_MSE = [MSE1, MSE2, MSE3, MSE4, MSE5]
    # plt.plot(list_layer,list_layer_MSE,marker='o', color='blue', linewidth=1.0, linestyle='--', label='linear line')
    # plt.xlabel('Number of Layers')
    # plt.ylabel('MSE')
    # my_x_ticks = np.arange(1, 6,1)
    # #my_y_ticks = np.arange(20, 70, 10)
    # plt.xticks(my_x_ticks)
    # #plt.yticks(my_y_ticks)
    # plt.show()
y_validation_pred = []
# print(x_test[x_test.shape[0]-30])
# x_validation = x_test[x_test.shape[0]-30:x_test.shape[0]-29]
# print(x_validation)
# y_v = rc_ELM(x_train, y_train, x_validation, C)
# x_validation[0][29] = y_v
# print(x_validation)
for i in range(14):
    x_validation = x_test[x_test.shape[0] - 14 + i:x_test.shape[0] - 13 + i]
    y_v = rc_ELM(x_train, y_train, x_validation, C)
    x_validation[0][29] = y_v
    y_validation_pred.append(y_v)
print(y_validation_pred)
x = range(14)

for i in range(len(y_validation_pred)):
    y_validation_pred[i] = y_validation_pred[i][0]
print(y_validation_pred)
plt.plot(x, y_validation, c='b', label='TEST')
plt.plot(x, y_validation_pred, c='r', label='RC_ELM')
plt.legend()
plt.show()

#     y_validation.append(y_v)
# print(y_validation)

#plt.plot(x,y_)