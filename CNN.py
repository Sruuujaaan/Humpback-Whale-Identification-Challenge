import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import random
import gc


# train_dir = "A:/Kaggle Datasets/whale-categorization-playground/train/train"
# test_dir = "A:/Kaggle Datasets/whale-categorization-playground/test/test"
#
# train_whales = ["A:/Kaggle Datasets/whale-categorization-playground/train/train/{}".format(i) for i in os.listdir(train_dir)]
# print(((train_whales)))


# print(train_whales)
train = pd.read_csv('train.csv')
# print(train.iloc[:, 1])
# print(len(train.iloc[:, 1].unique()))

a = train.iloc[: , 0]
b = "A:/Kaggle Datasets/whale-categorization-playground/train/train/"+a
train['Image'] = b

train_whales = (train['Image'].values)

# print(train_whales)
# print(train['Id'][0],'asf')
n_rows = 100
ncolumns = 100
channels = 3

def read_and_process_images(list_of_images):

    X = []
    y = []

    for i, image in enumerate(list_of_images):
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (n_rows,ncolumns) , interpolation = cv2.INTER_CUBIC ))
        y.append(train['Id'][i])
    return X,y

X, y = read_and_process_images(train_whales)

del train_whales
gc.collect()

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

print(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = y.reshape(-1, 1)
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()
print(len(y[0]))


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val  = train_test_split(X,y,test_size=.01, random_state = 2)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

del X
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)
print(ntrain,nval)
batch_size = 32

from keras.layers import Conv2D,MaxPooling2D, Flatten,Dense,Dropout
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

classifier.add(Dropout(.1))
# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 4251, activation = 'softmax'))

print(classifier.summary())
# Compiling the CNN
classifier.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
                                   shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train,y_train, batch_size = batch_size)
val_generator = val_datagen.flow(X_val,y_val, batch_size=batch_size)

history = classifier.fit_generator(train_generator,steps_per_epoch=ntrain//batch_size,epochs=400,
                                   validation_data=val_generator,validation_steps=nval//batch_size)

classifier.save('WhaleClassifier.h5')