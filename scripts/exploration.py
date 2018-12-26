import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as tdata
from sklearn.model_selection import train_test_split


#read train data. 
traindata = pd.read_csv('data/train.csv', dtype = np.float32)
traindata.head()

D_in, H, D_out = 784, 100, 10

x_train = traindata.iloc[:, 1: ].values
y_train =  traindata['label'].values


train_target = torch.from_numpy(y_train).type(torch.LongTensor)
train_features = torch.from_numpy(x_train)

traintensor = tdata.TensorDataset(train_features, train_target) 

trainloader = tdata.DataLoader(dataset = traintensor,
                                    batch_size = 45, 
                                    shuffle = True)

dataiter = iter(trainloader)
images, labels = dataiter.next()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x= F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

model = Net()

# specify the loss function. 
loss_fun = nn.NLLLoss()

# learning rate. 
lr = 1e-4
# optimizer. 
optimizer = torch.optim.Adam(model.parameters(), lr= lr)

for epoch in range(2):
    running_loss = 0

    for i, data in enumerate(trainloader, 0):
        
        features, labels = data
        features, labels = Variable(features), Variable(labels)
        features = features.view(-1, 28 * 28)
        labels = labels.view(-1)
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
