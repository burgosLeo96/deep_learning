import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential

#DATA LOADING ------------------------------------------------------------------------------------------------------------------------
nw_16_url = './data/NW2016.csv'
nw_17_url = './data/NW2017.csv'
nw_18_url = './data/NW2018.csv'
NW2016_dataset = pd.read_csv(nw_16_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
NW2017_dataset = pd.read_csv(nw_17_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)
NW2018_dataset = pd.read_csv(nw_18_url, header = 0, sep = ',', quotechar= '"', error_bad_lines = False)

print(NW2016_dataset.tail())


#DATASET PREPRoCESSING ---------------------------------------------------------------------------------------------------------------
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
dataset = dataset.drop(columns = ['lat', 'lon', 'height_sta', 'precip'], axis = 1)

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

TIMESTEP = '720T'
resample_ds = dataset.resample(TIMESTEP).mean()

train_ds = resample_ds.sample(frac=0.7)
test_ds = resample_ds.drop(train_ds.index)

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

HISTORY_LAG = 100
FUTURE_TARGET = 50

X_td_train, Y_td_train = segment(train_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)
X_td_train = X_td_train.reshape(X_td_train.shape[0], HISTORY_LAG, 1)
Y_td_train = Y_td_train.reshape(Y_td_train.shape[0], FUTURE_TARGET, 1)

print("Data shape: ", X_td_train.shape)
print("Tags shape: ", Y_td_train.shape)


X_td_test, Y_td_test = segment(test_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)

X_td_test = X_td_test.reshape(X_td_test.shape[0], HISTORY_LAG, 1)
Y_td_test = Y_td_test.reshape(Y_td_test.shape[0], FUTURE_TARGET, 1)

print("Data shape: ", X_td_test.shape)
print("Tags shape: ", Y_td_test.shape)

# MODEL TRAINING STAGE ------------------------------------------------------------------------------------------------------------------------

EPOCHS = 200

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(HISTORY_LAG, input_shape=X_td_train.shape[-2:]),
    tf.keras.layers.Dense(FUTURE_TARGET)
])

lstm_model.compile(optimizer='adam', metrics=['mae', 'mse'], loss='mse')

print('Now starting to train!')
tmstmp1 = time.time()

lstm_model.fit(X_td_train, Y_td_train, epochs=EPOCHS)

tmstmp2 = time.time()
print('Total time elapsed = ', tmstmp2 - tmstmp1)

# MODEL EVALUATION STAGE ------------------------------------------------------------------------------------------------------------------------
loss, mae, mse = lstm_model.evaluate(X_td_train, Y_td_train, verbose=2)

# MODEL PREDICTION STAGE ------------------------------------------------------------------------------------------------------------------------
predictions = lstm_model.predict(X_td_test, verbose = 0)

Y_td_test = Y_td_test.reshape(Y_td_test.shape[0], FUTURE_TARGET,)

predict_list = []
test_list = []

for i in predictions:
    predict_list.append(i[40])

pred_array = np.array(predict_list)
print(pred_array.shape)

for i in Y_td_test:
    test_list.append(i[40])

val_narray = np.array(test_list)
print(val_narray.shape)

a = plt.axes(aspect='equal')
plt.scatter(val_narray, pred_array)
plt.xlabel('True Values temperature')
plt.ylabel('Predictions temperature')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()

error = pred_array - val_narray
print(error)
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error temperature")
_ = plt.ylabel("Count")

plt.show()