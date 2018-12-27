import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from sklearn.model_selection import train_test_split
import pickle as pkl

#read train data. 
traindata = pd.read_csv('data/train.csv', 
                        dtype = np.float32)
traindata.head()

# split the data right now. 
y = traindata['label'].values
x = traindata.drop('label', axis = 1).values
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size= 0.4, random_state = 0
)

# save for later. 
train_ = {"x": xtrain, "y":ytrain}
test_ = {"x": xtest, "y": ytest}

with open("adhoc/train.pkl", "wb") as f:
    pkl.dump(train_, f)
f.close()
with open("adhoc/validate.pkl", "wb") as f:
    pkl.dump(test_, f)
f.close()

# preview the images first
plt.figure(figsize=(12,10))
x, y = 10, 4
for i in range(40):  
    plt.subplot(y, x, i+1)
    plt.imshow((xtrain[i]/255.0).reshape((28,28)),
                interpolation='nearest')
plt.show()

xtrain[0] / 255.0
