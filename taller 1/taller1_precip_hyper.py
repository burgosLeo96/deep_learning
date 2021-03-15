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
from tensorflow.keras.models import Sequential
from hyperas import optim
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

from functions import segment, normalize

def data():
    TIMESTEP = '720T'
    HISTORY_LAG = 100
    FUTURE_TARGET = 50
    nw_16_url = './data/NW2016.csv'
    nw_17_url = './data/NW2017.csv'
    nw_18_url = './data/NW2018.csv'

    NW2016_dataset = pd.read_csv(nw_16_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    NW2017_dataset = pd.read_csv(nw_17_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    NW2018_dataset = pd.read_csv(nw_18_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    
    data = pd.concat([NW2016_dataset, NW2017_dataset, NW2018_dataset])
    data = NW2016_dataset
    data = data[data.isna()['psl'] == False]
    #Drop useless columns
    data = data.drop(columns = ['lat', 'lon', 'height_sta'], axis = 1)

    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d %H:%M')
    data.set_index('date', inplace=True)
    
    #Interpolate missing values
    data = data.interpolate(method='linear')

    stats = data.describe()
    stats = stats.transpose()
    data = normalize(data, stats)

    resample_ds = data.resample(TIMESTEP).mean()

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
def model(X_train, X_test, y_train, y_test):

    model = tf.keras.models.Sequential()
    model.add(LSTM(units={{choice([100, 200, 500])}}, input_shape=X_train.shape[-2:]))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
    model.compile(optimizer='adam',
                       metrics=['mae', 'mse'], loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    model.fit(X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs={{choice([100, 200, 500, 1000])}},
              validation_split=0.08,
              callbacks=[early_stopping, checkpointer])
    
    loss, mae, mse  = model.evaluate(X_test, y_test, verbose=0)

    print('Test mae:', mae)
    print('Test mse:', mse)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)
    print(best_model)