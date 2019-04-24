import cv2
import numpy as np
import pandas as pd

import gc

train = pd.read_csv('train.csv')

a = train.iloc[: , 0]
b = "A:/Kaggle Datasets/whale-categorization-playground/train/train/"+a
train['Image'] = b

train_whales = (train['Image'].values)


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
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
with open('labekencodery.pickle', 'wb') as handle:
    pickle.dump(labelencoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

y = y.reshape(-1, 1)
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()
with open('onehotencoder.pickle', 'wb') as handle:
    pickle.dump(onehotencoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
