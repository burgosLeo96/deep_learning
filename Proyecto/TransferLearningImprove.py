import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd

train_df=pd.read_csv('./archive/dataset/train.csv')

print(train_df.head())
print(train_df['Class'].value_counts())

from keras.preprocessing import image


import matplotlib.pyplot as plt
import seaborn as sns

img=image.load_img('./archive/dataset/train/image1000.jpg')

img=image.img_to_array(img)/255
plt.imshow(img)
plt.show()
print(img.shape)

#Dividing data in valid and train part using StratifiedKFold as data is skewed
from sklearn.model_selection import StratifiedKFold
train_df['Kfold']=-1

print(train_df.head())


train_df=train_df.sample(frac=1).reset_index(drop=True)
print(train_df.tail())
y=train_df['Class']

kf=StratifiedKFold(n_splits=5)
for f,(t_,v_) in enumerate(kf.split(X=train_df,y=y)):
    train_df.loc[v_,'Kfold']=f

print(train_df.head())

train=train_df[train_df['Kfold']!=4]
valid=train_df[train_df['Kfold']==4]


print(valid.tail())

print(valid['Class'].value_counts())

#As the data is large so it will be better to use datagenerator, so I am using 
#keras Imagedatagenerator and we can do data augmentation in this step

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         width_shift_range=0.1,
#         height_shift_range=0.1)

train_generator=train_datagen.flow_from_dataframe(dataframe=train,
                                            directory="./archive/dataset/train/",
                                            x_col="Image",
                                            y_col="Class",
                                            subset="training",
                                            batch_size=128,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(331,331))

from keras.preprocessing.image import ImageDataGenerator

valid_datagen = ImageDataGenerator(rescale=1./255)
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         width_shift_range=0.1,
#         height_shift_range=0.1)

valid_generator=valid_datagen.flow_from_dataframe(dataframe=valid,
                                            directory="./archive/dataset/train/",
                                            x_col="Image",
                                            y_col="Class",
                                            subset="training",
                                            batch_size=128,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(331,331))

#Loading nasnet large model and setting all layers except last 35 as
#  non trainable so we can generalise our model on our data and also do transfer learning for better results

from keras.applications.nasnet import NASNetLarge
# from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Model
from tensorflow.keras.utils import to_categorical


resnet=NASNetLarge(include_top=True,weights='imagenet')
x=resnet.layers[-2].output
fc1=Dense(6,activation='softmax')(x)

my_model=Model(inputs=resnet.input,outputs=fc1)

print(my_model.summary())
from keras.optimizers import Adam
from keras import layers,models
adam=Adam(learning_rate=0.0001)

for l in my_model.layers[:-5]:
    #print(l)
    l.trainable = False
my_model.add(layers.Dense(512,activation='relu'))
my_model.add(layers.Dropout(0.5))
my_model.add(layers.Dense(256,activation='relu'))
my_model.add(layers.Dense(6,activation='softmax'))

my_model.compile(optimizer='adam',loss ="categorical_crossentropy",metrics=["accuracy"])

r = my_model.fit_generator(train_generator,steps_per_epoch=5176//128,validation_data=valid_generator,validation_steps=1293//128,epochs=2)

my_model.save('model.h5')

#Ploting Loss ansd Accuracy

# Loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss_TransferLearning')

# Accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc_TransferLearning')

import os
name=[]
y_pred=[]

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())


import numpy as np

s=0
for i in os.listdir('./archive/dataset/test/'):
    name.append(i)
    i='./archive/dataset/test/'+i
    img=image.load_img(i,target_size=(331,331,3))
    img=image.img_to_array(img)/255
    pred=my_model.predict(img.reshape(1,331,331,3))
    y_pred.append(labels[np.argmax(pred[0])])
    s+=1
    if s%100==0:
        print(s)

data=pd.DataFrame((zip(name,y_pred)),columns=['Image','Class'])

print(data.head())

data.to_csv('submission.csv',index=False)

print(data.shape)