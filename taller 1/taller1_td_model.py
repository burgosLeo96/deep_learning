import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from random import random
import tensorflow as tf
import csv

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from hyperas import optim
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation

from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


#dd	Wind direction	degrees (°)
#ff	Wind speed	m.s-1
#precip	Precipitation during the reporting period	kg.m2
#hu	Humidity	percentage (%)
#td	Dew point	Kelvin (K)
#t	Temperature	Kelvin (K)
#psl	Pressure reduced to sea level	Pascal (Pa)



# NORMALIZE DATASET 
def normalize(x, stats):
    return (x - stats['mean']) / stats['std']



#print(dataset.tail())

# DATASET SEGMENTATION ------------------------------------------------------------------------------------------------------------------------

TIMESTEP = '720T'
HISTORY_LAG = 100
FUTURE_TARGET = 50
DENSE_NEURONS = 64
LEARNING_RATE = 0.01
NN_ARCHITECTURE = 0
EPOCHS = [500, 1000, 2000, 4000, 6000]

headers = ['ITERATION', 'TIMESTEP', 'HISTORY_LAG', 'DENSE_NEURONS', 'LEARNING_RATE', 'FUTURE_TARGET', 'TRANING_EPOCHS', 'NN_ARCHITECTURE', 'LOSS', 'MAE', 'MSE']
report = []
report.append(headers)

def load_data():
    nw_16_url = './data/NW2016.csv'
    nw_17_url = './data/NW2017.csv'
    nw_18_url = './data/NW2018.csv'

    NW2016_dataset = pd.read_csv(nw_16_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    NW2017_dataset = pd.read_csv(nw_17_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    NW2018_dataset = pd.read_csv(nw_18_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    
    data = pd.concat([NW2016_dataset, NW2017_dataset, NW2018_dataset])
    data = data[data.isna()['psl'] == False]
    #Drop useless columns
    data = data.drop(columns = ['lat', 'lon', 'height_sta', 'precip'], axis = 1)

    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d %H:%M')
    data.set_index('date', inplace=True)
    
    #Interpolate missing values
    data = data.interpolate(method='linear')

    stats = data.describe()
    stats = stats.transpose()
    data = normalize(data)
    return data


def data(dataset):
    maxlen = 100
    max_features = 20000
    iteration = 1
    resample_ds = dataset.resample(TIMESTEP).mean()

    train_ds = resample_ds.sample(frac=0.7)
    test_ds = resample_ds.drop(train_ds.index)

    X_train, y_train = segment(train_ds, "precip", window = HISTORY_LAG, future = FUTURE_TARGET)
    X_train = X_train.reshape(X_train.shape[0], HISTORY_LAG, 1)
    y_train = y_train.reshape(y_train.shape[0], FUTURE_TARGET, 1)

    X_test, y_test = segment(train_ds, "precip", window = HISTORY_LAG, future = FUTURE_TARGET)
    X_test = X_test.reshape(X_test.shape[0], HISTORY_LAG, 1)
    y_test = y_test.reshape(y_test.shape[0], FUTURE_TARGET,)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, X_test, y_train, y_test

#Model the NN
def model(X_train, X_test, y_train, y_test, max_features, maxlen):

    model = tf.keras.models.Sequential()
    model.add(LSTM(HISTORY_LAG, input_shape=input.shape[-2:]))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    
    model.compile(optimizer='adam',
                       metrics=['mae', 'mse'], loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    model.fit(X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              nb_epoch={{choice([100, 200, 500, 1000])}},
              validation_split=0.08,
              callbacks=[early_stopping, checkpointer])
    
    score, acc = model.evaluate(X_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def segment(dataset, variable, window = 5000, future = 0):
    data = []
    labels = []
    for i in range(len(dataset)):
        start_index = i
        end_index = i + window
        future_index = i + window + future
        if future_index >= len(dataset):
            break
        data.append(dataset[variable][i:end_index])
        labels.append(dataset[variable][end_index:future_index])
    return np.array(data), np.array(labels)

if __name__ == '__main__':
    
    
    
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)
    

# MODEL PREDICTION STAGE ------------------------------------------------------------------------------------------------------------------------
#predictions = lstm_model.predict(X_td_test, verbose = 0)

#Y_td_test = Y_td_test.reshape(Y_td_test.shape[0], FUTURE_TARGET,)

#predict_list = []
#test_list = []

#for i in predictions:
#    predict_list.append(i[40])

#pred_array = np.array(predict_list)
#print(pred_array.shape)

#for i in Y_td_test:
#    test_list.append(i[40])

#val_narray = np.array(test_list)
#print(val_narray.shape)

#a = plt.axes(aspect='equal')
#plt.scatter(val_narray, pred_array)
#plt.xlabel('True Values temperature')
#plt.ylabel('Predictions temperature')
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)

#plt.show()

#error = pred_array - val_narray
#print(error)
#plt.hist(error, bins = 25)
#plt.xlabel("Prediction Error temperature")
#_ = plt.ylabel("Count")

#plt.show()
#Opening CSV file to save results
with open('./reports/td_model_report.csv', mode = 'w') as td_results:
    writer = csv.writer(td_results)
    writer.writerows(report)