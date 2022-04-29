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
    Test the NN model or retrain from scratch, see if __main__
    In any case, run via python3 nn.py --model nnX for X = 20,50,100
"""
def ReLU(X):
    relu = np.vectorize(lambda x: x * (x > 0))
    return relu(X)

def standard_eval(X,Y,serialized,layers):
    conv = np.array(serialized['conv']['weight']['w'])
    bias_conv = np.array(serialized['conv']['bias']['b'])
    dense, bias_dense = [],[]
    for i in range(layers):
        dense.append(np.array(serialized[f'dense_{str(i+1)}']['weight']['w']))
        bias_dense.append(np.array(serialized[f'dense_{str(i+1)}']['bias']['b']))
    
    CONV, CONV_BIAS = torch.from_numpy(conv).double(), torch.from_numpy(bias_conv).double()
    X = F.conv2d(X, CONV, CONV_BIAS, stride=1, padding=1)
    X = F.relu(X)
    X = X.reshape(X.shape[0], -1)
    for d,b in zip(dense, bias_dense):
        D,B = torch.from_numpy(d).double(), torch.from_numpy(b).double()
        X = F.relu(F.linear(X, D, B))
    
    _, predicted_labels = X.max(1)
    corrects = (predicted_labels == Y).sum().item()

    return corrects

def linear_eval(X,Y, serialized,layers):
    """
        Linear pipeline without normalization and regular relu works fine
        Problem is when introducing relu_approx which need normalization to stay within interval
    """
    #conv = np.array(serialized['conv']['weight']['w'])
    #bias_conv = np.array(serialized['conv']['bias']['b'])
    #CONV, CONV_BIAS = torch.from_numpy(conv).double(), torch.from_numpy(bias_conv).double()
    #exp = F.conv2d(X, CONV, CONV_BIAS, stride=1, padding=1)
    
    X = F.pad(X, [1,1,1,1])
    X = X.reshape(X.shape[0],-1)
    
    print("mean of input:",X.mean())
    
    conv, convMT = pack_conv_rect(np.array(serialized['conv']['weight']['w']), 10,11,1,30,30)
    bias_conv = pack_bias(np.array(serialized['conv']['bias']['b']), 2, 840//2)

    dense, bias_dense = [],[]
    for i in range(layers):
        dense.append(np.array(serialized[f'dense_{str(i+1)}']['weight']['w']))
        bias_dense.append(np.array(serialized[f'dense_{str(i+1)}']['bias']['b']).reshape(1,-1))
    conv, conv_bias = convMT, np.array(bias_conv['b']).reshape(1,-1)
    conv, conv_bias = preprocessing.normalize(conv), preprocessing.normalize(conv_bias)
    
    X = X @ conv
    for i in range(len(X)):
        X[i] += conv_bias
    #X = relu_approx(X)
    X = ReLU(X)
    
    for d,b in zip(dense, bias_dense):
        d,b = preprocessing.normalize(d), preprocessing.normalize(b)
        X = X @ d.T
        for i in range(len(X)):
            ## needed for reasons...
            try:
                X[i] = np.array(X[i]) + b 
            except:
                X[i] = X[i] + b
        
        for x in X.flatten():
            if x > interval or x < -interval:
                print("Outside interval:", x)
        #X = relu_approx(X)
        X = ReLU(X)

        #print("mean after activation:", X.mean()) #ok if normalized

    pred = np.argmax(X,axis=1)
    #print("Results: ", pred)
    #print("Expected labels: ", Y)
    corrects = np.sum(pred == Y.numpy())

    return corrects

def test_pipeline(eval):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="simplenet, nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    
    with open(f'./models/{args.model}.json', 'r') as f:
        serialized = json.load(f)
    if args.model == "nn20":
        layers = 20
    elif args.model == "nn50":
        layers = 50
    elif args.model == "nn100":
        layers = 100
    batchsize = 1000
    dh = DataHandlerNN("./data/mnist_validation", batchsize)
    corrects = 0.0
    tot = 0
    for batch in dh.batch():
        x = np.array(batch[0]).reshape(batchsize,1,28,28)
        X = torch.from_numpy(x)
        y = np.array(batch[1]).reshape(batchsize)
        Y = torch.from_numpy(y)
        corrects += eval(X.double(),Y.double(),serialized,layers)
        tot += batchsize
    print("Accuracy:")
    print(corrects/tot)

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

    dataHandler = DataHandler(dataset="MNIST", batch_size=256)
    model = NN(layers)
    logger = Logger("./logs/",f"nn{layers}")
    model.apply(model.weights_init)
    train(logger, model, dataHandler, num_epochs=10, lr=0.001)
    loss, accuracy = eval(logger, model, dataHandler)

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
    #train_nn_from_scratch()
    
    ## use to evaluate the standard nn model with conv and relu
    print("[*] Standard eval:")
    test_pipeline(standard_eval)    

    ## use to evaluate the model with a linearized conv and approximated relus
    print("[*] Linear eval:")
    test_pipeline(linear_eval)