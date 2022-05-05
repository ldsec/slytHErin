from pkg_resources import yield_lines
from pytest import yield_fixture
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import json
import argparse
import numpy as np
from nn import NN
from activation import *
from logger import Logger
from dataHandler import *
from utils import *
from conv_transform import *
from activation import *
"""
    Script for testing nn with approximations for homomorphic encryption
"""
os.chdir(".")
# explicit function to normalize array
def normalize(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def ReLU(X):
    relu = np.vectorize(lambda x: x * (x > 0))
    return relu(X)

def linear_eval(X,Y, serialized):
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
    X = X.numpy()

    conv = np.array(serialized['conv']['weight']['w']).reshape(serialized['conv']['weight']['rows'],serialized['conv']['weight']['cols'])
    conv_bias = np.array(serialized['conv']['bias']['b'])
    dense, bias = [],[]
    for d in serialized['dense']:
        dense.append(np.array(d['weight']['w']).reshape(d['weight']['rows'], d['weight']['cols']))
        bias.append(np.array(d['bias']['b']))
    X = X @ conv
    for i in range(len(X)):
        X[i] += conv_bias
    X = ReLU(X)
    
    iter = 0
    for d,b in zip(dense, bias):
        X = X @ d
        for i in range(len(X)):
            X[i] = X[i] + b
        
        #for x in X.flatten():
        #    if x > interval or x < -interval:
        #        print("Outside interval:", x)
        if iter != len(dense)-1:
            X = ReLU(X)
        iter += 1

    pred = np.argmax(X,axis=1)
    corrects = np.sum(pred == Y.numpy())

    return corrects

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    device = 'cpu'
    if "nn" not in args.model:
        print("call --model nn20, nn50 or nn100")
        exit()
    with open(f'./models/{args.model}_packed.json', 'r') as f:
        serialized = json.load(f)
    if args.model == "nn20":
        layers = 20
        model = torch.load("./models/nn20.pt")
    elif args.model == "nn50":
        layers = 50
        model = torch.load("./models/nn50.pt")
    elif args.model == "nn100":
        layers = 100
        model = torch.load("./models/nn100.pt")

    model = model.to(device)

    batchsize = 512
    dataHandler = DataHandler(dataset="MNIST", batch_size=batchsize)
    corrects = 0
    corrects_torch = 0
    tot = 0.0
    for X,Y in dataHandler.test_dl:
        corrects += linear_eval(X.double(),Y.double(),serialized)
        tot += batchsize
        predictions = model(X)
        _,predicted_labels = predictions.max(1)
        corrects_torch += (predicted_labels == Y).sum().item()
    print("Accuracy:")
    print(corrects/tot)
    print("Expected:")
    print(corrects_torch/tot)
