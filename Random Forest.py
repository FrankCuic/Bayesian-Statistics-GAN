import os
import math
import pandas as pd
import numpy as np
import openpyxl
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
from prettytable import PrettyTable
warnings.filterwarnings("ignore")

dataset=pd.read_csv(r"C:\Users\TyrantFrank\Desktop\stock.csv",encoding='gb2312')

print(dataset)

values = dataset.values[:,1:]

values = values.astype('float32')

def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples):
    res = np.zeros((num_samples,n_in*or_dim+n_out))
    for i in range(0, num_samples):
        h1 = values[scroll_window*i: n_in+scroll_window*i,0:or_dim]
        h2 = h1.reshape( 1, n_in*or_dim)
        h3 = values[n_in+scroll_window*(i) : n_in+scroll_window*(i)+n_out,-1].T
        h4 = h3[np.newaxis, :]
        h5 = np.hstack((h2,h4))
        res[i,:] = h5
    return res


n_in = 5
n_out = 2
or_dim = values.shape[1]
num_samples = 2000
scroll_window = 1
res = data_collation(values, n_in, n_out, or_dim, scroll_window, num_samples)
values = np.array(res)

n_train_number = int(num_samples * 0.85)
Xtrain = values[:n_train_number, :n_in*or_dim]
Ytrain = values[:n_train_number, n_in*or_dim:]


Xtest = values[n_train_number:, :n_in*or_dim]
Ytest = values[n_train_number:,  n_in*or_dim:]

# 对训练集和测试集进行归一化
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test = m_out.transform(Ytest)

model = RandomForestRegressor(n_estimators=60, max_features='sqrt')
model.fit(vp_train, vt_train)


yhat = model.predict(vp_test)
yhat = yhat.reshape(num_samples-n_train_number, n_out)

predicted_data = m_out.inverse_transform(yhat)


def mape(y_true, y_pred):
    record = []
    for index in range(len(y_true)):
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        record.append(temp_mape)
    return np.mean(record) * 100

def evaluate_forecasts(Ytest, predicted_data, n_out):
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    table = PrettyTable(['test set index','MSE', 'RMSE', 'MAE', 'MAPE','R2'])
    for i in range(n_out):
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        if n_out == 1:
            strr = 'prediction result index：'
        else:
            strr = 'No.'+ str(i + 1)+'step prediction result index：'
        table.add_row([strr, mse, rmse, mae, str(MApe)+'%', str(r2*100)+'%'])

    return mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table



mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, predicted_data, n_out)

print(table)

from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False
         }
rcParams.update(config)

plt.ion()
for ii in range(n_out):

    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 2), dpi=300)
    x = range(1, len(predicted_data) + 1)
    plt.xticks(x[::int((len(predicted_data)+1))])
    plt.tick_params(labelsize=5)
    plt.plot(x, predicted_data[:,ii], linestyle="--",linewidth=0.5, label='predict')
    plt.plot(x, Ytest[:,ii], linestyle="-", linewidth=0.5,label='Real')

    plt.rcParams.update({'font.size': 5})

    plt.legend(loc='upper right', frameon=False)

    plt.xlabel("Sample points", fontsize=5)

    plt.ylabel("value", fontsize=5)

    if n_out == 1:
        plt.title(f"The prediction result of RANDOM-Forest :\nMAPE: {mape(Ytest[:, ii], predicted_data[:, ii])} %")
    else:
        plt.title(f"{ii+1} step of RANDOM-Forest prediction\nMAPE: {mape(Ytest[:,ii], predicted_data[:,ii])} %")
    # plt.xlim(xmin=600, xmax=700)

    # plt.savefig('figure/prediction result.png')

plt.ioff()
plt.show()