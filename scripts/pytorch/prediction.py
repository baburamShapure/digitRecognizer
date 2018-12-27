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


model  = torch.load("models/20181226_181317")

def make_submission(model):
    test_data = pd.read_csv("data/test.csv", dtype= np.float32)
    testTensor = torch.from_numpy(test_data.values).view(-1, 28 * 28)
    outputs = model(testTensor)
    pred = torch.max(outputs.data, 1)[1]
    pred.numpy()
    outd = pd.DataFrame({'ImageId': range(1, test_data.shape[0]+1),
                        'Label': pred.numpy()})
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    outd.to_csv(os.path.join('out', timestamp + '.csv'), 
                            index = None)


make_submission(model)
