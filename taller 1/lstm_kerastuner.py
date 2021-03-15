from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
import numpy as np
import time

import kerastuner as kt

#dd	Wind direction	degrees (°)
#ff	Wind speed	m.s-1
#precip	Precipitation during the reporting period	kg.m2
#hu	Humidity	percentage (%)
#td	Dew point	Kelvin (K)
#t	Temperature	Kelvin (K)
#psl	Pressure reduced to sea level	Pascal (Pa)

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

# NORMALIZE DATASET ------------------------------------------------------------------------------------------------------------------------

def normalize(x):
    return (x - stats['mean']) / stats['std']

dataset = normalize(dataset)

# DATASET SEGMENTATION ------------------------------------------------------------------------------------------------------------------------

#Assign some initial values to the hyperparams
TIMESTEP = '720T'
HISTORY_LAG = 100
DENSE_NEURONS = 64
LEARNING_RATE = 0.01
FUTURE_TARGET = 1
EPOCHS = [500, 1000, 2000, 4000, 6000]

#Defining segment function
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

resample_ds = dataset.resample(TIMESTEP).mean()

train_ds = resample_ds.sample(frac=0.7)
test_ds = resample_ds.drop(train_ds.index)

X_precip_train, Y_precip_train = segment(train_ds, "precip", window = HISTORY_LAG, future = FUTURE_TARGET)
X_precip_train = X_precip_train.reshape(X_precip_train.shape[0], HISTORY_LAG, 1)
Y_precip_train = Y_precip_train.reshape(Y_precip_train.shape[0], FUTURE_TARGET, 1)

X_precip_test, Y_precip_test = segment(test_ds, "precip", window = HISTORY_LAG, future = FUTURE_TARGET)
X_precip_test = X_precip_test.reshape(X_precip_test.shape[0], HISTORY_LAG, 1)
Y_precip_test = Y_precip_test.reshape(Y_precip_test.shape[0], FUTURE_TARGET, 1)

print('X_train shape:', X_precip_train.shape)
print('X_test shape:', X_precip_test.shape)
print('Finished create_data method')

def create_model(hp):
    print('Started create_model method')    

    model = Sequential()
    model.add(LSTM(units=hp.Int('units',
                                min_value=32,
                                max_value=512,
                                step=32),
                    activation='relu', 
                    input_shape=X_precip_train.shape[-2:]))
    model.add(Dense(units=hp.Int('units',
                        min_value=32,
                        max_value=512,
                        step=32),
                    activation='relu'))
    model.add(Dense(FUTURE_TARGET))

    model.compile(loss='sme',
                  optimizer='adam',
                  metrics=['accuracy', 'mae', 'mse'])

    print('Finished create_model method')  

    return model

tuner = kt.Hyperband(
    create_model, objective="val_accuracy", max_epochs=30, hyperband_iterations=2
)

tuner.search_space_summary()

tuner.search(
    X_precip_train, Y_precip_train,
    validation_data=(X_precip_test, Y_precip_test),
    callbacks=[EarlyStopping(patience=1)],
)

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

print(best_model)
print(best_hyperparameters)
