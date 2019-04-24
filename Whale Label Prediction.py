from keras.models import load_model
import pickle
classifier = load_model('WhaleClassifier.h5')
images = []
whalelabels = []
with open('labekencodery.pickle', 'rb') as handle:
    tokenize6 = pickle.load(handle)

with open('onehotencoder.pickle', 'rb') as handle:
    tokenize7 = pickle.load(handle)

import numpy as np
from keras.preprocessing import image
import os
for root in os.walk("A:/Kaggle Datasets/whale-categorization-playground/test/"):
    for dir in root[2:] :
        for files in dir:
            images.append(files)
            location = str("A:/Kaggle Datasets/whale-categorization-playground/test/test/"+files)
            test_image = image.load_img(location, target_size = (100, 100))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = classifier.predict(test_image, verbose=1)


            for i, pred in enumerate(result):
                print(" ".join(tokenize6.inverse_transform(pred.argsort()[-5:][::-1].reshape(-1,1)).tolist()))
                whalelabels.append((tokenize6.inverse_transform(pred.argsort()[-5:][::-1].reshape(-1,1))))

            # print(" ".join(x) for x in whalelabels)
            # whalelabels.append(y_pred.argsort()[-5:][::-1])
            # print(y_pred.argsort()[-5:][::-1],'-----------------')

print(images,whalelabels)


import pandas as pd
names = [name for name in images]
whalelabelss = []
print(whalelabels)

for elems in whalelabels:
    whalelabelss.append(' '.join(elems))

sub = pd.DataFrame({'Image': names, 'Id': whalelabelss}, columns=['Image', 'Id'])
sub.to_csv('whale.csv', index=False)


