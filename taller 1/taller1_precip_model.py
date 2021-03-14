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

#dd	Wind direction	degrees (°)
#ff	Wind speed	m.s-1
#precip	Precipitation during the reporting period	kg.m2
#hu	Humidity	percentage (%)
#td	Dew point	Kelvin (K)
#t	Temperature	Kelvin (K)
#psl	Pressure reduced to sea level	Pascal (Pa)

#Drop useless columns
dataset = dataset.drop(columns = ['lat', 'lon', 'height_sta', 'td'], axis = 1)

dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d %H:%M')
dataset.set_index('date', inplace=True)

#Interpolate missing values
dataset = dataset.interpolate(method='linear')

stats = dataset.describe()
stats = stats.transpose()

#print(dataset.head())
#print(dataset.tail())

# NORMALIZE DATASET ------------------------------------------------------------------------------------------------------------------------

def normalize(x):
    return (x - stats['mean']) / stats['std']

dataset = normalize(dataset)

#print(dataset.tail())

# DATASET SEGMENTATION ------------------------------------------------------------------------------------------------------------------------

#Assign some initial values to the hyperparams
TIMESTEP = '720T'
HISTORY_LAG = 100
DENSE_NEURONS = 64
LEARNING_RATE = 0.01
FUTURE_TARGET = 50
EPOCHS = 200
NN_ARCHITECTURE = 0

report = ['ITERATION', 'TIMESTEP', 'HISTORY_LAG', 'DENSE_NEURONS', 'LEARNING_RATE', 'FUTURE_TARGET', 'TRANING_EPOCHS', 'NN_ARCHITECTURE', 'LOSS', 'MAE', 'MSE']

#Defining two possible NN architectures
def lstm_opc1(input):

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(HISTORY_LAG, input_shape=input.shape[-2:]),
        tf.keras.layers.Dense(FUTURE_TARGET)
    ])

    lstm_model.compile(optimizer='adam', metrics=['mae', 'mse'], loss='mse')

    return lstm_model

def lstm_opc2(input):
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(HISTORY_LAG, input_shape=input.shape[-2:]),
        tf.keras.layers.Dense(DENSE_NEURONS),
        tf.keras.layers.Dense(FUTURE_TARGET)
    ])

    optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)

    lstm_model.compile(optimizer=optimizer, metrics=['mae', 'mse'], loss='mse')
    return lstm_model

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

for x in range(10):
    print("iteration ", x)
    resample_ds = dataset.resample(TIMESTEP).mean()

    train_ds = resample_ds.sample(frac=0.7)
    test_ds = resample_ds.drop(train_ds.index)

    X_precip_train, Y_precip_train = segment(train_ds, "precip", window = HISTORY_LAG, future = FUTURE_TARGET)
    X_precip_train = X_precip_train.reshape(X_precip_train.shape[0], HISTORY_LAG, 1)
    Y_precip_train = Y_precip_train.reshape(Y_precip_train.shape[0], FUTURE_TARGET, 1)

    #print("Data shape: ", X_precip_train.shape)
    #print("Tags shape: ", Y_precip_train.shape)


    X_precip_test, Y_precip_test = segment(test_ds, "precip", window = HISTORY_LAG, future = FUTURE_TARGET)

    X_precip_test = X_precip_test.reshape(X_precip_test.shape[0], HISTORY_LAG, 1)
    Y_precip_test = Y_precip_test.reshape(Y_precip_test.shape[0], FUTURE_TARGET, 1)

    #print("Data shape: ", X_precip_test.shape)
    #print("Tags shape: ", Y_precip_test.shape)

    # MODEL TRAINING STAGE ------------------------------------------------------------------------------------------------------------------------

    #print('Now starting to train!')
    tmstmp1 = time.time()
    lstm_model = None

    rnd = random()
    
    if(rnd <= 0.5):
        NN_ARCHITECTURE = 1
        lstm_model = lstm_opc1(X_precip_train)
    else:
        NN_ARCHITECTURE = 2
        lstm_model = lstm_opc2(X_precip_train)

    lstm_model.fit(X_precip_train, Y_precip_train, epochs=EPOCHS)

    tmstmp2 = time.time()
    print('Total time elapsed = ', tmstmp2 - tmstmp1)

    # MODEL EVALUATION STAGE ------------------------------------------------------------------------------------------------------------------------
    loss, mae, mse = lstm_model.evaluate(X_precip_train, Y_precip_train, verbose=2)
    report.append([x, TIMESTEP, HISTORY_LAG, DENSE_NEURONS, LEARNING_RATE, FUTURE_TARGET, EPOCHS, NN_ARCHITECTURE, loss, mae, mse])

# MODEL PREDICTION STAGE ------------------------------------------------------------------------------------------------------------------------
#predictions = lstm_model.predict(X_precip_test, verbose = 0)

#Y_precip_test = Y_precip_test.reshape(Y_precip_test.shape[0], FUTURE_TARGET,)

#predict_list = []
#test_list = []

#for i in predictions:
#    predict_list.append(i[40])

#pred_array = np.array(predict_list)
#print(pred_array.shape)

#for i in Y_precip_test:
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
#plt.hist(error, bins = 25)
#plt.xlabel("Prediction Error temperature")
#_ = plt.ylabel("Count")

#plt.show()
#Opening CSV file to save results
with open('./reports/precip_model_report.csv', mode = 'w') as precip_results:
    writer = csv.writer(precip_results)
    writer.writerows(report)