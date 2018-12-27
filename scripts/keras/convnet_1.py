import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
import gc
import pickle as pkl 
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import datetime as dt
import os

train_ = pkl.load(open("adhoc/train.pkl", 'rb'))
xtrain = train_["x"]
ytrain =  train_["y"]
# normalize.
xtrain = xtrain / 255.0

# reshape
xtrain = xtrain.reshape(-1, 28, 28, 1)

# one hot encoding of target. 
ytrain_ohe = keras.utils.to_categorical(ytrain, 10)

# VGG like net. 

model = Sequential()
# input: 28X28 images with 3 channels
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), 
                activation='relu', 
                input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3),  
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(xtrain, ytrain_ohe, batch_size=32, epochs=10)

score = model.evaluate(x_test, y_test, batch_size=32)
