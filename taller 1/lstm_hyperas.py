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

def create_data():
    TIMESTEP = '720T'
    
    nw_16_url = './data/NW2016.csv'
    nw_17_url = './data/NW2017.csv'
    nw_18_url = './data/NW2018.csv'

    print('Loading data...')

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

    resample_ds = dataset.resample(TIMESTEP).mean()

    train_ds = resample_ds.sample(frac=0.7)
    test_ds = resample_ds.drop(train_ds.index)

    HISTORY_LAG = 100
    FUTURE_TARGET = 50

    print("Started data segmentation")

    data_train = []
    labels_train = []
    for i in range(len(train_ds)):
        start_index = i
        end_index = i + HISTORY_LAG
        future_index = i + HISTORY_LAG + FUTURE_TARGET
        if future_index >= len(train_ds):
            break
        data_train.append(train_ds['precip'][i:end_index])
        labels_train.append(train_ds['precip'][end_index:future_index])

    X_precip_train = np.array(data_train)
    Y_precip_train = np.array(labels_train)

    X_precip_train = X_precip_train.reshape(X_precip_train.shape[0], HISTORY_LAG, 1)
    Y_precip_train = Y_precip_train.reshape(Y_precip_train.shape[0], HISTORY_LAG, 1)

    data_test = []
    labels_test = []
    for i in range(len(test_ds)):
        start_index = i
        end_index = i + HISTORY_LAG
        future_index = i + HISTORY_LAG + FUTURE_TARGET
        if future_index >= len(test_ds):
            break
        data_test.append(test_ds['precip'][i:end_index])
        labels_test.append(test_ds['precip'][end_index:future_index])

    X_precip_test = np.array(data_test)
    Y_precip_test = np.array(labels_test)

    X_precip_test = X_precip_test.reshape(X_precip_test.shape[0], HISTORY_LAG, 1)
    Y_precip_test = Y_precip_test.reshape(Y_precip_test.shape[0], FUTURE_TARGET, 1)

    print('X_train shape:', X_precip_train.shape)
    print('X_test shape:', X_precip_test.shape)
    print('Finished create_data method')
    return X_precip_train, X_precip_test, Y_precip_train, Y_precip_test, HISTORY_LAG, FUTURE_TARGET


def create_model(X_precip_train, X_precip_test, Y_precip_train, Y_precip_test, HISTORY_LAG, FUTURE_TARGET):
    print('Started create_model method')    

    model = Sequential()
    model.add(LSTM(HISTORY_LAG, input_shape=X_precip_train.shape[-2:]))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(FUTURE_TARGET))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    
    print("Starting process fit step")
    tmstmp1 = time.time()

    model.fit(X_precip_train, Y_precip_train,
              batch_size={{choice([32, 64, 128])}},
              validation_split=0.08,
              callbacks=[early_stopping])

    tmstmp2 = time.time()
    print('Total time elapsed = ', tmstmp2 - tmstmp1)

    score, acc = model.evaluate(X_precip_test, Y_precip_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=create_data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)
