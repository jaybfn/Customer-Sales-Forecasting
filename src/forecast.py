import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import LSTM ,Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from keras.regularizers import L1L2
from tensorflow.keras.models import load_model
from numpy.random import seed

# moving average calculation
def moving_average_(dataframe, window_size):
    
    #window_size = 4
    #tot = sales['Total].tolist()
    if window_size != 1:
        i = 0
        moving_averages = []
        while i < len(dataframe['Total']) - window_size + 1:
            this_window = dataframe['Total'][i : i + window_size]
            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += 1

        sales = pd.DataFrame(moving_averages, columns=['Total'])

    else: 
        moving_averages = dataframe['Total']
        sales = pd.DataFrame(moving_averages, columns=['Total'])
    return sales

#adding lag to the sales for multistep forecasting
def lags(dataframe, lags):
    for lag in range(1,lags):
        col_name = 'lag_' +str(lag)
        dataframe[col_name] = dataframe['sales_norm'].shift(lag)
    #drop null val
    dataframe = dataframe.dropna().reset_index(drop = True)
    return dataframe

def forecasting(dataframe, window_size, lag, n_future):
    
    #dataframe = sales_
    #window_size = 4
    #lags = 6
    #n_future = number of weeks /days to forecast in future

    model = load_model(r'..\models\model_final.h5') 
    scaler = MinMaxScaler()
    for n in range(n_future):
    
        sales = pd.DataFrame(moving_average_(dataframe, window_size), columns=['Total'])
        sales_norm = scaler.fit_transform(sales.Total.values.reshape(-1, 1))
        sales_norm = sales_norm.flatten().tolist()
        sales['sales_norm'] = sales_norm
        sales = sales.drop(['Total'],axis = 1)
        sales = lags(sales,lag)
        last_row = sales[-1:]
        last_row = last_row.drop(['sales_norm'], axis = 1)
        last_row = last_row.to_numpy()
        last_row = last_row.reshape(last_row.shape[0], 1, last_row.shape[1])
        pred = model.predict(last_row)
        forecast = scaler.inverse_transform(pred)[:,0]#.tolist()
        forecast = pd.DataFrame(forecast, columns = ['Total'])
        dataframe = pd.concat([dataframe, forecast], ignore_index=True)
        subset = pd.DataFrame(dataframe.tail(n_future))
    plt.plot(np.arange(n_future),subset['Total'])
    plt.show()
    return subset

sales_tot = pd.read_csv('../data/sales_full.csv', index_col=0)
#sales_tot.head()
sales_tot = sales_tot.drop(['year','week','weeks'], axis = 1)
sales_ = sales_tot.copy()

data = forecasting(sales_,5, 6, 12)



