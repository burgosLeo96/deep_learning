import pathlib
import time

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#Load prediction data

NW2016_dataset = pd.read_csv('./NW2016.csv')
NW2017_dataset = pd.read_csv('./NW2017.csv')
NW2018_dataset = pd.read_csv('./NW2018.csv')

join_dataset = pd.concat([NW2016_dataset, NW2017_dataset, NW2018_dataset])

#for dirname, _, filenames in os.walk('/kaggle/input'):
#for filename in filenames:
#print(os.path.join(dirname, filename))

join_dataset = join_dataset[join_dataset.isna()['psl'] == False]

# Define the labels from the variable/columns that will be used. 
features_considered = ['dd', 'ff', 'precip', 'hu','td', 't', 'psl']

#Drop useless columns
join_dataset.drop(columns = ['number_sta', 'lat', 'lon', 'height_sta', 'date'], inplace = True)

#Interpolate missing values
join_dataset = join_dataset.interpolate(method='linear')


print(join_dataset.tail())

#Variables to predict: precip, t

