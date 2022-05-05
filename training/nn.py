from pkg_resources import yield_lines
from pytest import yield_fixture
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import json
import argparse
import numpy as np
from activation import *
from logger import Logger
from dataHandler import *
from utils import *
from conv_transform import *
from activation import *

"""
    Train NN
    run via python3 nn.py --model nnX for X = 20,50,100
"""

class NN(nn.Module):
  '''
    Retraining of ZAMA NN architecture for inputs and weights in [-10,10]
  '''
  
  def __init__(self, layers, verbose = False):
    super().__init__()
    self.verbose = verbose
    self.pad = F.pad
    self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(10,11), stride=1, device=device)
    dense = []
    dense.append(nn.Linear(840,92,True, device=device))
    for _ in range(layers-2):
        dense.append(nn.Linear(92,92,True,device=device))
    dense.append(nn.Linear(92,10,True,device=device))
    self.dense = nn.ModuleList(dense)
    #safety check
    assert(len(self.dense)==layers)
    
    self.activation = nn.ReLU()
    #self.activation = ReLUApprox()

  def forward(self, x):
    x = self.pad(x, (1,1,1,1))
    x = self.conv(x)
    x = self.activation(x)
    x = x.reshape(x.shape[0],-1) #flatten
    
    for i,layer in enumerate(self.dense):
        x = layer(x)
        if i != len(self.dense)-1:
            x = self.activation(x)

    return x
 
  def weights_init(self, m):
    for m in self.children():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)

def train_nn_from_scratch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="simplenet, nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    
    if args.model == "nn20":
        layers = 20
    elif args.model == "nn50":
        layers = 50
    elif args.model == "nn100":
        layers = 100

    dataHandler = DataHandler(dataset="MNIST", batch_size=512)
    model = NN(layers)
    logger = Logger("./logs/",f"nn{layers}")
    model.apply(model.weights_init)
    start = time.time()
    train_advanced(logger, model, dataHandler, num_epochs=50, lr=0.001)
    end = time.time()
    print("--- %s seconds for train---" % (end - start))
    loss, accuracy = eval_advanced(logger, model, dataHandler)

    torch.save(model, f"./models/nn{layers}.pt")
    
    ##store as json for Go
    dense = []
    bias = []
    for name, p in model.named_parameters():
        if "conv" in name:
            if "weight" in name:
                conv = p.data.cpu().numpy()
            else:
                bias_conv = p.data.cpu().numpy()
        else:
            if "weight" in name:
                dense.append(p.data.cpu().numpy())
            else:
                bias.append(p.data.cpu().numpy())
    serialized = serialize_nn(conv,bias_conv,dense,bias,layers+1)
    packed = pack_nn(serialized,layers)
    
    with open(f'./models/{args.model}_packed.json', 'w') as f:
        json.dump(packed, f)
    
if __name__ == '__main__':
    ## use for training a new model
    train_nn_from_scratch()