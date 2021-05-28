from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("./archive/dataset/train.csv")

print(df.head())

print(df['Class'].value_counts())

df['Class'].value_counts(sort=True).plot.bar()

plt.show()

train_image = './archive/dataset/train'
test_image = './archive/dataset/test'

#Setting Input Image Size
image_size = [299, 299]

#Splitting Dataframe

train_df,valid_df = train_test_split(df,test_size=.15,stratify=df.Class.values,shuffle=True)


train_df.reset_index(inplace=True,drop=True)
valid_df.reset_index(inplace=True,drop=True)

#Training Dataframe

print(train_df.head())
train_df['Class'].value_counts()


train_df['Class'].value_counts(sort=True).plot.bar()
plt.show()


#Validation Dataframe
print(valid_df.head())

valid_df['Class'].value_counts()

valid_df['Class'].value_counts(sort=True).plot.bar()
plt.show()

#Getting Images and Preprocesing
#Getting Images for training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_image,
                                                 x_col="Image", y_col="Class",
                                                 class_mode="categorical",
                                                 target_size=(299,299), batch_size=32)
#Getting Images for Vaidation

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_set = validation_datagen.flow_from_dataframe(dataframe=valid_df, directory=train_image,
                                                 x_col="Image", y_col="Class",
                                                 class_mode="categorical",
                                                 target_size=(299,299), batch_size=32)

#Getting The InceptionResNetV2 Model
incep = InceptionResNetV2(input_shape=image_size + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in incep.layers:
  layer.trainable = False  

x = Flatten()(incep.output)


prediction = Dense(6, activation='softmax')(x)

# create a model object
model = Model(inputs=incep.input, outputs=prediction)

# view the structure of the model
model.summary()                                                 

#Compile the model

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#Fitting the model

r = model.fit(
  training_set,
  validation_data=validation_set,
  epochs=12,
  steps_per_epoch=training_set.n//32,
  validation_steps=validation_set.n//32
)

#Ploting Loss ansd Accuracy

# Loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#Getting Predictions for Test Data Images

name=[]
y_pred=[]
labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
for i in os.listdir('./archive/dataset/test/'):
    name.append(i)
    i='./archive/dataset/test/'+i
    img=image.load_img(i,target_size=(299,299,3))
    img=image.img_to_array(img)/255
    pred=model.predict(img.reshape(1,299,299,3))
    y_pred.append(labels[np.argmax(pred[0])])
    
data=pd.DataFrame((zip(name,y_pred)),columns=['Image','Class'])
data.head()

data.to_csv('submission_Inception.csv',index=False)

data.shape