import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from functions import normalize

def data():
    #Load training data
    raw_dataset = pd.read_csv('/content/WindTurbine_Dataset_Hourly-weather-obs_ChurchLawford_2015_Telemetry.csv')

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

    dataset.tail()

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
    normed_train_data = normalize(train_dataset)
    normed_test_data = normalize(test_dataset)
    normed_full_data = normalize(full_dataset)
    normed_full_data.tail()

def build_model():
      model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# Inspect the model. Use the .summary method to print a simple description of the model.
model.summary()

# Now try out the model with model.predict to check it is consistent before we try lots
# of epochs!
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

def build_model():
          model = keras.Sequential([
            layers.Dense(nn, activation=act, input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
          ])
          optimizer = tf.keras.optimizers.RMSprop(0.001)
          model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model

        model = build_model()

        # Inspect the model. Use the .summary method to print a simple description of the model.
        model.summary()

        # Now try out the model with model.predict to check it is consistent before we try lots
        # of epochs!
        example_batch = normed_train_data[:10]
        example_result = model.predict(example_batch)
        example_result


        """
        ================================================================
        Train the model for a max. number of epochs, but put a stagnaion
        condition if improvements stop well before.
        """

        EPOCHS = 250

        # Restart the model if needed for another training test
        # model = build_model()

        # The patience parameter is the amount of epochs to check for improvement

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        print('Now starting to train!')
        tmstmp1 = time.time()
        # Call the model training function.
        history = model.fit(normed_train_data, train_labels, 
            epochs=EPOCHS, validation_split = 0.2, verbose=1,
                callbacks=[early_stop])

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss with act function %s' %act)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        tmstmp2 = time.time()
        print('Total time elapsed = ', tmstmp2 - tmstmp1)

        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
        results.append([act, nn, loss, mae])

        print("Testing set Mean Abs Error with activation function",act, ": {:5.2f} Power_Out_KW".format(mae))