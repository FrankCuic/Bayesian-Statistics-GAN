import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math

## import data
df = pd.read_csv('DATA.csv', parse_dates=['Date'])
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)

# Create Apple stock price plot
## https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df['Date'], df['Close'], label='Apple stock')
ax.set(xlabel="Date",
       ylabel="USD",
       title="Apple Stock Price")
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()


# Calculate technical indicators
def get_technical_indicators(data):
    # Create 7 and 21 days Moving Average
    data['MA7'] = data.iloc[:, 4].rolling(window=7).mean()
    data['MA21'] = data.iloc[:, 4].rolling(window=21).mean()

    # Create MACD
    data['MACD'] = data.iloc[:, 4].ewm(span=26).mean() - data.iloc[:, 1].ewm(span=12, adjust=False).mean()

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 4].rolling(20).std()
    data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:, 4].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:, 4] - 1)

    return data


T_df = get_technical_indicators(df)

# Drop the first 21 rows
# For doing the fourier
dataset = T_df.iloc[30:, :].reset_index(drop=True)


# Getting the Fourier transform features
def get_fourier_transform(dataset):
    data_FT = dataset[['Date', 'Close']]

    close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
    n = len(close_fft)

    # Initialize DataFrame to store components
    fft_com_df = pd.DataFrame()

    # Determine the unique components
    num_unique_components = n // 2 + 1

    for num_ in [10, 20, 30]:
        # Create a mask for positive frequencies
        mask = np.zeros(n, dtype=bool)
        mask[:num_] = True  # Set the first num_ components to True

        # If the signal length is odd, we don't duplicate the highest frequency component
        if n % 2:  # n is odd
            mask[-num_ + 1:] = True  # Set the last num_-1 components to True
        else:  # n is even
            mask[-num_:] = True  # Set the last num_ components to True

        # Apply mask to the FFT coefficients
        fft_list_m10 = np.copy(close_fft)
        fft_list_m10[~mask] = 0  # Zero out all other components

        # Perform inverse FFT
        ifft_result = np.fft.ifft(fft_list_m10)

        # Store the absolute values and angles of the result
        fft_com_df[f'absolute_{num_}_comp'] = np.abs(ifft_result)
        fft_com_df[f'angle_{num_}_comp'] = np.angle(ifft_result)

    return fft_com_df


# Get Fourier features
dataset_F = get_fourier_transform(dataset)
Final_data = pd.concat([dataset, dataset_F], axis=1)

print(Final_data.head())

Final_data.to_csv("Finaldata_with_Fourier.csv", index=False)


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['MA7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Close'], label='Closing Price', color='b')
    plt.plot(dataset['MA21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Apple - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['logmomentum'], label='Momentum', color='b', linestyle='-')

    plt.legend()
    plt.show()


plot_technical_indicators(T_df, 400)


def plot_fourier_transforms(dataset):
    data_FT = dataset['Close']
    close_fft = np.fft.fft(data_FT.values)
    n = len(close_fft)

    # Prepare the frequency bins for plotting (not necessary for the calculation)
    fft_freq = np.fft.fftfreq(n)  # Get the corresponding frequencies
    print("FFT Frequency bins:", fft_freq)

    # Use plt.figure() once, and then add to this figure for each subsequent plot
    plt.figure(figsize=(14, 7))

    # Plot the original 'Close' data
    plt.plot(dataset['Date'], dataset['Close'], label='Real')

    for num_ in [10, 20, 30]:
        # Initialize a mask for all components as False
        mask = np.zeros(n, dtype=bool)
        # Set the first num_ components and the last num_ components as True
        mask[:num_] = True
        if n % 2 == 0:  # if even
            mask[-num_:] = True
        else:  # if odd
            mask[-(num_ - 1):] = True

        fft_list_m10 = np.copy(close_fft)
        fft_list_m10[~mask] = 0

        ifft_result = np.fft.ifft(fft_list_m10)
        plt.plot(dataset['Date'], np.real(ifft_result), label=f'Fourier {num_} components')

    plt.title(f'Fourier Transform with 3, 6, 9 Components')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.show()



# Assuming `dataset` is your DataFrame with a 'Close' column
# Plot the Fourier transform
plot_fourier_transforms(dataset)
