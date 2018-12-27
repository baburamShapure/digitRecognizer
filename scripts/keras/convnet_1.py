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

#  serialize model to JSON
timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
model_json = model.to_json()
with open("models/keras/model_" + timestamp +".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("models/keras/model_" + timestamp + ".h5")
print("Saved model to disk")

# the validation datasets. 
validate_  = pkl.load(open("adhoc/validate.pkl", "rb"))
xvalidate = validate_["x"]
yvalidate = validate_["y"]
xvalidate = xvalidate / 255.0
yvalidate_ohe = keras.utils.to_categorical(yvalidate, 10)
xvalidate = xvalidate.reshape(-1, 28, 28, 1)

model.evaluate(x= xvalidate, y= yvalidate_ohe)

y_pred = model.predict(xvalidate)
y_pred_classes = np.argmax(y_pred, axis = 1) 
confusion_mtx = confusion_matrix(yvalidate, y_pred_classes) 


# predict on test data. 
testdata = pd.read_csv("data/test.csv", dtype= np.float32)
xtest = testdata.values/255.0
xtest = xtest.reshape(-1, 28, 28, 1)
ytest_scores = model.predict(xtest)
ytest_class = np.argmax(ytest_scores, axis= 1)

outd = pd.DataFrame({'ImageId': range(1, testdata.shape[0]+1),
                        'Label': ytest_class})

timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
outd.to_csv(os.path.join('out', timestamp + '.csv'), 
                            index = None)

