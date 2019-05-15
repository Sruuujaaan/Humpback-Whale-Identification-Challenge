import cv2
import numpy as np
import pandas as pd

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

print(len(X),y)
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# We have 2 inputs, 1 for each picture
left_input = Input((100,100,3))
right_input = Input((100,100,3))

# We will use 2 instances of 1 network for this task
convnet = Sequential([
    Conv2D(5,3, input_shape=(100,100,3)),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(5,3),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(7,2),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(7,2),
    Activation('relu'),
    Flatten(),
    Dense(18),
    Activation('sigmoid')
])
# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

image_list = X
label_list = y


left_input = []
right_input = []
targets = []

# Number of pairs per image
pairs = 5
# Let's create the new dataset to train on
for i in range(len(label_list)):
    for _ in range(pairs):
        compare_to = i
        while compare_to == i:  # Make sure it's not comparing to itself
            compare_to = random.randint(0, 999)
        left_input.append(image_list[i])
        right_input.append(image_list[compare_to])
        if label_list[i] == label_list[compare_to]:  # They are the same
            targets.append(1.)
        else:  # Not the same
            targets.append(0.)

left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))

siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=10,
          verbose=1,
          validation_data=([left_input, right_input], targets)
                )

siamese_net.save('WhaleClassifierSia.h5')




