import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./archive/dataset/train.csv')
df.head()

from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

#Using flow from dataframe method for mapping dataframe and directory both

train_generator = datagen.flow_from_dataframe(
    df,
    directory='./archive/dataset/train',
    x_col = 'Image',
    y_col = 'Class',
    target_size=(299,299),
    class_mode = 'categorical',
    batch_size=32)


#Using InceptionV3 pretrained model
base_model = InceptionV3(include_top=False,weights='imagenet',input_shape=(299,299,3))

base_model.trainable = False

from keras import layers,models
#Adding some extra layers over pretrained model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))

