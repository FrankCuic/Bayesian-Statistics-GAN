import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from statsmodels.graphics.tsaplots import plot_acf

dataset = pd.read_csv('Finaldata_with_Fourier.csv', parse_dates=['Date'])

dataset.replace(0, np.nan, inplace=True)

dataset.ffill(inplace=True)
dataset.bfill(inplace=True)

dataset.set_index('Date', inplace=True)
dataset.sort_index(inplace=True)

print("Columns in dataset:", dataset.columns)

target_column = 'Close'
X_values = dataset.drop(columns=[target_column])
y_values = dataset[target_column]

plot_acf(y_values, lags=100)
plt.show()

# Normalize Data
X_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit and transform the data
X_scaled = X_scaler.fit_transform(X_values)
y_scaled = y_scaler.fit_transform(y_values.values.reshape(-1, 1))

dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(y_scaler, open('y_scaler.pkl', 'wb'))

input_window = 30
output_window = 1

def reshape_time_series_data(X, y, input_window, output_window):
    X_reshaped, y_reshaped = [], []
    for i in range(len(X) - input_window - output_window + 1):
        X_reshaped.append(X[i:(i + input_window)])
        y_reshaped.append(y[(i + input_window):(i + input_window + output_window)])
    return np.array(X_reshaped), np.array(y_reshaped)

X_reshaped, y_reshaped = reshape_time_series_data(X_scaled, y_scaled.flatten(), input_window, output_window)

def split_data(X, y, train_ratio=0.7):
    train_size = int(len(X) * train_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X_reshaped, y_reshaped)

# %% --------------------------------------- Save Processed Data -------------------------------------------------------
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
