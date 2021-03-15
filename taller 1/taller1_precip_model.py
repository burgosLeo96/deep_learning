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

from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

#dd	Wind direction	degrees (°)
#ff	Wind speed	m.s-1
#precip	Precipitation during the reporting period	kg.m2
#hu	Humidity	percentage (%)
#td	Dew point	Kelvin (K)
#t	Temperature	Kelvin (K)
#psl	Pressure reduced to sea level	Pascal (Pa)

def create_model(X_precip_train, Y_precip_train, X_precip_test, Y_precip_test):
    # MODEL TRAINING STAGE ------------------------------------------------------------------------------------------------------------------------

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM({{choice([100])}}, input_shape=X_precip_train.shape[-2:]),
        tf.keras.layers.Dropout({{uniform(0, 1)}}),
        tf.keras.layers.Dense({{choice([50])}})
    ])

    lstm_model.compile(
        optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
        metrics=['mae', 'mse', 'accuracy'], loss='mse'
    )

    #print('Now starting to train!')
    tmstmp1 = time.time()

    lstm_model.fit(
    X_precip_train, 
    Y_precip_train,
    batch_size={{choice([32, 64, 128])}},
    epochs={{choice[500, 1000, 2000]}}, 
    verbose = 0)

    tmstmp2 = time.time()
    print('Total time elapsed = ', tmstmp2 - tmstmp1)

    # MODEL EVALUATION STAGE ------------------------------------------------------------------------------------------------------------------------
    loss, mae, mse, accuracy = lstm_model.evaluate(X_precip_train, Y_precip_train, verbose=2)

    print("Loss: ", loss)
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("Acc: ", accuracy)
    
    return lstm_model

def data():
    #DATA LOADING ------------------------------------------------------------------------------------------------------------------------
    nw_16_url = './data/NW2016.csv'
    nw_17_url = './data/NW2017.csv'
    nw_18_url = './data/NW2018.csv'

    print("Started data loading process")

    NW2016_dataset = pd.read_csv(nw_16_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    NW2017_dataset = pd.read_csv(nw_17_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
    NW2018_dataset = pd.read_csv(nw_18_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)

    print("Data loaded successfully")

    #DATASET PREPROCESSING ---------------------------------------------------------------------------------------------------------------
    dataset = pd.concat([NW2016_dataset, NW2017_dataset, NW2018_dataset])
    dataset = dataset[dataset.isna()['psl'] == False]

    #Drop useless columns
    dataset = dataset.drop(columns = ['lat', 'lon', 'height_sta', 'td'], axis = 1)

    dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d %H:%M')
    dataset.set_index('date', inplace=True)

    #Interpolate missing values
    dataset = dataset.interpolate(method='linear')

    stats = dataset.describe()
    stats = stats.transpose()

    dataset = (dataset - stats['mean']) / stats['std']

    # DATASET SEGMENTATION ------------------------------------------------------------------------------------------------------------------------

    resample_ds = dataset.resample('720T').mean()

    train_ds = resample_ds.sample(frac=0.7)
    test_ds = resample_ds.drop(train_ds.index)

    #window = {{choice([100])}}
    #future = {{choice([50])}}

    window = 100
    future = 50

    print("Started data segmentation")

    data_train = []
    labels_train = []
    for i in range(len(train_ds)):
        start_index = i
        end_index = i + window
        future_index = i + window + future
        if future_index >= len(train_ds):
            break
        data_train.append(train_ds['precip'][i:end_index])
        labels_train.append(train_ds['precip'][end_index:future_index])

    X_precip_train = np.array(data_train)
    Y_precip_train = np.array(labels_train)

    X_precip_train = X_precip_train.reshape(X_precip_train.shape[0], window, 1)
    Y_precip_train = Y_precip_train.reshape(Y_precip_train.shape[0], future, 1)

    data_test = []
    labels_test = []
    for i in range(len(test_ds)):
        start_index = i
        end_index = i + window
        future_index = i + window + future
        if future_index >= len(test_ds):
            break
        data_test.append(test_ds['precip'][i:end_index])
        labels_test.append(test_ds['precip'][end_index:future_index])

    X_precip_test = np.array(data_test)
    Y_precip_test = np.array(labels_test)

    X_precip_test = X_precip_test.reshape(X_precip_test.shape[0], window, 1)
    Y_precip_test = Y_precip_test.reshape(Y_precip_test.shape[0], future, 1)

    print(len(X_precip_train), 'train sequences')
    print(len(X_precip_test), 'test sequences')

    
    print('X_train shape:', X_precip_train.shape)
    print('X_test shape:', X_precip_test.shape)

    return X_precip_train, Y_precip_train, X_precip_test, Y_precip_test

def main():
    #Assign some initial values to the hyperparams
    TIMESTEP = '720T'
    HISTORY_LAG = 100
    DENSE_NEURONS = 64
    LEARNING_RATE = 0.01
    FUTURE_TARGET = 50
    EPOCHS = [500, 1000, 2000]
    NN_ARCHITECTURE = 0

    best_run, best_model = optim.minimize(
        data=data,
        model=create_model,
        algo=tpe.suggest,
        max_evals=10,
        trials = Trials())

    print(best_run)

main()