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

class LSTM_Model():
    # setting the seed for BI-LSTM network so that everytime you run the network the prediction remains the same!
    seed(42)
    tf.random.set_seed(42) 
    # 
    def __init__(self, dataframe, window_size, lags, trainsize):

        self.dataframe = dataframe
        self.window_size = window_size
        self.lags = lags
        self.trainsize = trainsize

    # moving average calculation
    def scaled_moving_average(self):

        scaler = MinMaxScaler()
        moving_averages = []
        if self.window_size != 1:
            i = 0
            while i < len(self.dataframe['Total']) - self.window_size + 1:
                this_window = self.dataframe['Total'][i : i + self.window_size]
                window_average = sum(this_window) / self.window_size
                moving_averages.append(window_average)
                i += 1
            sales = pd.DataFrame(moving_averages, columns=['Total'])
            sales_norm = scaler.fit_transform(sales.Total.values.reshape(-1, 1))
            sales['sales_norm'] = sales_norm
            sales = sales.drop(['Total'],axis = 1)
        
        else:
            moving_averages = self.dataframe['Total']
            sales = pd.DataFrame(moving_averages, columns=['Total'])
            sales_norm = scaler.fit_transform(sales.Total.values.reshape(-1, 1))
            sales['sales_norm'] = sales_norm
            sales = sales.drop(['Total'],axis = 1)
        return sales

    #adding lag to the sales for multistep forecasting
    def lags_cal(self):

        sales = self.scaled_moving_average()
        for lag in range(1,self.lags):
            col_name = 'lag_' +str(lag)
            sales[col_name] = sales['sales_norm'].shift(lag)
        #drop null val
        sales = sales.dropna().reset_index(drop = True)
        return sales

    # tain and test dataset
    def split_train_test(self):

        """ This function splits the dataframe in to train and test sets and converts in to LSTM readable format
        It needs 2 input: 
            1. Dataframe to split the dta into train and test
            2. trainsize in percentage ratio.
                eg: if you want 80% of the data as training then plug in 0.8"""

        sales = self.lags_cal()
        train = sales[: int(len(sales)*self.trainsize)].values
        test =  sales[int(len(sales)*self.trainsize):].values
        X_train = train[:, 1:]
        y_train = train[:, 0:1]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = test[:, 1:]
        y_test = test[:, 0:1]
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        return X_train, y_train, X_test, y_test

    def build_LSTM(self):

        X_train, y_train, X_test, y_test = self.split_train_test()
        K.clear_session()
        model = Sequential()
        model.add(Bidirectional(LSTM(64, activation = 'tanh', 
                                input_shape = ( X_train.shape[1], X_train.shape[2]), 
                                return_sequences=True)))
        model.add(Bidirectional(LSTM(32, activation = 'tanh', 
                                return_sequences = False)))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation = 'tanh'))
        model.add(Dense(y_train.shape[1]))

        return model

    # fit the model
    def fit_evaluate_model(self):
        
        """ This function prints all the epochs, loss and val score plot and also
        evaluation score on test data and return the fitmodel as 'mod', 
        which also can be used in to function (evaluate_model)"""

        X_train, y_train, X_test, y_test = self.split_train_test()
        mod = self.build_LSTM()
        mod.compile(optimizer='adam', loss='mse')
        cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                    min_delta=0.01,
                                    patience=150,
                                    verbose=1,
                                    mode="min",
                                    baseline=None,
                                    restore_best_weights=False)
        history = mod.fit(X_train,y_train, 
                    epochs = 150, 
                    batch_size = 8, 
                    validation_split=0.2, 
                    verbose = 1,
                    callbacks=[cb],
                    shuffle= True)
        mod.save(r"..\models\model_trail.h5")
        pd.DataFrame(history.history).plot()
        plt.grid(True)
        plt.gca() # set the y range to [0,1]
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        print('\n')
        print('*****************************************')
        print('\n')
        print('Model Evalution Score')
        print(mod.evaluate(X_test, y_test)) 
        return mod
        
    # model evaluation
    def evaluate_model(self,mod):

        X_train, y_train, X_test, y_test = self.split_train_test()
        print(mod.evaluate(X_test, y_test)) 
        
    """ How to run:
    sales_pred = LSTM_Model(sales_, 4, 6, 0.9)
    model = sales_pred.fit_evaluate_model()

    the below code is not necessary: 
    
    sales_pred.evaluate_model(model)   """



sales_tot = pd.read_csv('../data/sales_full.csv', index_col=0)
#sales_tot.head()
sales_tot = sales_tot.drop(['year','week','weeks'], axis = 1)
sales_ = sales_tot.copy()

sales_pred = LSTM_Model(sales_,4, 6, 0.9)
model = sales_pred.fit_evaluate_model()
