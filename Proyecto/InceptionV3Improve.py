import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.inception_resnet_v2 import InceptionResNetV2

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
base_model = InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(299,299,3))

base_model.trainable = False

from keras import layers,models
#Adding some extra layers over pretrained model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(768,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))


model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

train_steps = np.ceil(train_generator.n/train_generator.batch_size)


r = model.fit(
    train_generator,
    epochs=12,
    batch_size=32,
    steps_per_epoch=train_steps)
#Crucial step : Generated features of imagesby predicting it by removing the last layer of the model.


import keras
new_train_x = []
new_train_y = []
model2 = keras.Model(model.input, model.layers[-5].output)
count = 0
while count < 200:
    x_batch,y_batch = next(train_generator)
    pred = model2.predict(x_batch)
    new_train_x.extend(pred)
    new_train_y.extend(y_batch)
    count += 1


new_train_y = np.argmax(new_train_y,axis=1)
print(new_train_y.shape)

new_train_x = np.array(new_train_x)
new_train_y = np.array(new_train_y)
print(new_train_x.shape)
print(new_train_y.shape)

#Fitting new_train_x and new_train_y with xgboost

from xgboost import XGBClassifier
clf = XGBClassifier(max_depth=7, objective='multi:softmax', n_estimators=1000, 
                        num_classes=6)
clf.fit(new_train_x,new_train_y)

test_df = pd.DataFrame()
test_images = os.listdir('./archive/dataset/test')
test_df['Image']=test_images
test_df.head()


#Preparing test generator
test_generator = datagen.flow_from_dataframe(
    test_df,
    directory='./archive/dataset/test',
    x_col = 'Image',
    y_col = None,
    target_size=(299,299),
    class_mode = None,
    batch_size=32,
    shuffle = False)

#Predicting on test_generator

new_test_x = model2.predict(test_generator)
new_test_x = np.array(new_test_x)
predictions_xgb = clf.predict(new_test_x)


print(predictions_xgb)

test_df['Class']=predictions_xgb


num_to_class = dict((y,x) for (x,y) in train_generator.class_indices.items())
print(num_to_class)

test_df['Class']=test_df['Class'].map(num_to_class)
test_df.head()

test_df.to_csv('pred.csv',index=False)

