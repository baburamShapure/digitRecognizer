import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import gc
import pickle as pkl 
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import datetime as dt
import os


def reset():
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
    gc.collect()

reset()

train_ = pkl.load(open("adhoc/train.pkl", 'rb'))

xtrain = train_["x"]
ytrain =  train_["y"]

# normalize.
xtrain = xtrain / 255.0
ytrain = keras.utils.to_categorical(ytrain, 10)

model = Sequential()
model.add(Dense(64, activation = 'relu', input_dim = 28 * 28))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# adam optimizer with default parameters. 
optim = Adam()
model.compile(loss= 'categorical_crossentropy',
             optimizer = optim,
            metrics = ['accuracy'])
model.fit(xtrain, ytrain, epochs = 5, batch_size= 128)

# the validation datasets. 
validate_  = pkl.load(open("adhoc/validate.pkl", "rb"))
xvalidate = validate_["x"]
yvalidate = validate_["y"]
xvalidate = xvalidate / 255.0
yvalidate_ohe = keras.utils.to_categorical(yvalidate, 10)


model.evaluate(x= xvalidate, y= yvalidate_ohe)

# serialize model to JSON
timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
model_json = model.to_json()
with open("models/keras/model_" + timestamp +".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("models/keras/model_" + timestamp + ".h5")
print("Saved model to disk")


y_pred = model.predict(xvalidate)
y_pred_classes = np.argmax(y_pred, axis = 1) 
confusion_mtx = confusion_matrix(yvalidate, y_pred_classes) 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(confusion_mtx, range(10))


# predict on test data. 
testdata = pd.read_csv("data/test.csv", dtype= np.float32)
xtest = testdata.values/255.0

ytest_scores = model.predict(xtest)
ytest_class = np.argmax(ytest_scores, axis= 1)

outd = pd.DataFrame({'ImageId': range(1, testdata.shape[0]+1),
                        'Label': ytest_class})

timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
outd.to_csv(os.path.join('out', timestamp + '.csv'), 
                            index = None)

