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
import datetime as dt

# read data. 
fulldata = pd.read_csv("data/train.csv", dtype = np.float32)

xs = fulldata.loc[:, fulldata.columns != 'label'].values
ys = fulldata['label'].values

# train test split. 
xtrain, xtest, ytrain, ytest = train_test_split(xs, ys, test_size= 0.3, 
                                                random_state = 43)

def makeloader(x, y, bs):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).type(torch.LongTensor)
    __ = tdata.TensorDataset(x, y)
    dataloader = tdata.DataLoader(dataset = __,
                                    batch_size = bs, 
                                    shuffle = False)
    return(dataloader)

trainloader = makeloader(xtrain, ytrain, 40)
testloader = makeloader(xtest, ytest, 40)

# specify model. 
class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
    def forward(self, x):
        x= F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = simpleNet()
# specify loss
loss_fun = nn.CrossEntropyLoss()
# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)

N_EPOCHS = 5
loss_list = []
iteration_list = []
accuracy_list = [] 
count = 0


for t in range(N_EPOCHS):

    running_loss = 0
    
    for i, data in enumerate(trainloader, 0):
        features, target = data
        features = Variable(features)
        target = Variable(target)

        features = features.view(-1, 28 * 28)
        output = model(features)
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in testloader:
                images = Variable(images.view(-1, 28 * 28))
                
                # Forward propagation
                outputs = model(images)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))
        

timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
torch.save(model, os.path.join('models', timestamp))

