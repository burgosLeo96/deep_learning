import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation
from tensorflow import keras
from tensorflow.keras import layers
from functions import normalize
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from keras.optimizers import RMSprop
from functions import segment, normalize

def data():
    HISTORY_LAG = 500
    FUTURE_TARGET = 50
    #Load training data
    raw_dataset = pd.read_csv('./data/WindTurbine_Dataset_Hourly-weather-obs_ChurchLawford_2015_Telemetry.csv')

    # Define the labels from the variable/columns that will be used. 
    features_considered = ['wind_speed', 'stn_pres', 'air_temperature', 'rltv_hum','Power_Out_KW']

    # Create a clean Data Frame with only the data required
    dataset_copy = raw_dataset.copy()
    dataset = dataset_copy[features_considered]
    # dataset.tail()

    # The dataset contains a few unknown values.
    dataset.isna().sum()

    # So drop them if its the easiest (or put ave./default values if possible?)
    dataset = dataset.dropna()

    full_dataset = dataset.copy()
    train_dataset = dataset.sample(frac=0.7,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
      # Also look at the overall statistics:
    train_stats = train_dataset.describe()
    train_stats.pop('Power_Out_KW')
    train_stats = train_stats.transpose()
    train_mean = train_stats['mean']
    train_std = train_stats['std']

    print('Train data statistics', train_stats)

    # Separate the target value, or "label", from the features. This label is the value that you will train the model to predict.
    train_labels = train_dataset.pop('Power_Out_KW')
    test_labels = test_dataset.pop('Power_Out_KW')
    full_labels = full_dataset.pop('Power_Out_KW')
    normed_train_data = normalize(train_dataset,train_stats )
    stats = test_dataset.describe()
    stats = stats.transpose()
    normed_test_data = normalize(test_dataset,stats )
    stats = full_dataset.describe()
    stats = stats.transpose()
    normed_full_data = normalize(full_dataset, stats)
    
    return normed_train_data, normed_test_data, train_labels, test_labels

def model(normed_train_data, normed_test_data, train_labels, test_labels):
    
    model = keras.Sequential()
    model.add(Dense(64, input_shape=[len(normed_train_data.keys())] ))
    model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([10, 64, 100])}}))
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))
    model.add(Dense(1))
    
    rms = RMSprop()
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    
    # The patience parameter is the amount of epochs to check for improvement

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
          
    model.fit(normed_train_data, train_labels,
              batch_size={{choice([64, 128])}},
              epochs={{choice([100, 200, 500, 1000])}},
              verbose={{choice([1,2])}},
              validation_split=0.2,
              callbacks=[early_stop])
    
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print('Test mae:', mae)
    print('Test loss:', loss)
    print('Test mse:', mse)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    
    normed_train_data, normed_test_data, train_labels, test_labels = data()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print('best_run:', best_run)
    print('best_model:', best_model)
    print("Evalutation of best performing model:")
    print(best_model.evaluate(normed_test_data, test_labels))