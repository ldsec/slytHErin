import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import argparse
import numpy as np
from activation import *
from logger import Logger
from dataHandler import *
from utils import *
from conv_transform import *
from activation import *

os.chdir("./models")
"""
    Test the NN model
"""
def ReLU(X):
    relu = np.vectorize(lambda x: x * (x > 0))
    return relu(X)

def standard_eval(X,Y,serialized):
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

def linear_eval(X,Y, serialized):
    X = F.pad(X, [1,1,1,1])
    X = X.reshape(X.shape[0],-1)
    
    conv, convMT = pack_conv_rect(np.array(serialized['conv']['weight']['w']), 10,11,1,30,30)
    bias_conv = pack_bias(np.array(serialized['conv']['bias']['b']), 2, 840//2)
    dense, bias_dense = [],[]
    for i in range(layers):
        dense.append(np.array(serialized[f'dense_{str(i+1)}']['weight']['w']))
        bias_dense.append(np.array(serialized[f'dense_{str(i+1)}']['bias']['b']))
    
    conv, conv_bias = convMT, np.array(bias_conv['b'])
    X = X @ conv + conv_bias.T
    X = relu_approx(X)
    for d,b in zip(dense, bias_dense):
        X = X @ d.T + b.T
        X = relu_approx(X)

    pred = np.argmax(X,axis=1)
    corrects = np.sum(pred == Y.numpy())

    return corrects

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="simplenet, nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    
    with open(f'{args.model}.json', 'r') as f:
        serialized = json.load(f)
    if args.model == "nn20":
        layers = 20
    elif args.model == "nn50":
        layers = 50
    elif args.model == "nn100":
        layers = 100
    batchsize = 64
    dh = DataHandlerNN("../data/mnist_validation", batchsize)
    corrects = 0.0
    tot = 0.0
    for batch, fin in dh.batch():
        if fin:
            break
        x = np.array(batch[0]).reshape(batchsize,1,28,28)
        X = torch.from_numpy(x)
        y = np.array(batch[1]).reshape(batchsize)
        Y = torch.from_numpy(y)
        corrects += linear_eval(X.double(),Y.double(),serialized)
        tot += batchsize
    print("Accuracy:")
    print(corrects/tot)


    
    ##TO DO: save the results of the standard pipeline and test the linearized one