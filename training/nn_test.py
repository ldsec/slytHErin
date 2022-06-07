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
#Max value in NN50 = 7

os.chdir(".")

# explicit function to normalize array
def normalize(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def relu_np(X):
    relu = np.vectorize(lambda x: x * (x > 0))
    return relu(X)

def relu_approx_np(X):
    X = torch.from_numpy(X)
    X = relu_approx(X)
    return X.numpy()

def silu_np(X):
    return X/(1 + np.exp(-X))

def silu_approx_np(X):
    X = torch.from_numpy(X)
    X = X * sigmoid_approx(X)
    X = X.numpy()
    return X

def soft_relu_np(X):
    s = np.vectorize(lambda x: math.log(1+math.exp(x)))
    return s(X)

def soft_relu_approx_np(X):
    X = torch.from_numpy(X)
    X = softrelu_approx(X)
    return X.numpy()

def linear_eval(X,Y, serialized):
    """
        Evaluates the HE friendly pipeline
    """
    #conv = np.array(serialized['conv']['weight']['w'])
    #bias_conv = np.array(serialized['conv']['bias']['b'])
    #CONV, CONV_BIAS = torch.from_numpy(conv).double(), torch.from_numpy(bias_conv).double()
    #exp = F.conv2d(X, CONV, CONV_BIAS, stride=1, padding=1)
    
    #X = F.pad(X, [1,1,1,1])
    X = X.reshape(X.shape[0],-1)
    X = X.numpy()

    #record max values seen
    max = 0.0

    conv = np.array(serialized['conv']['weight']['w']).reshape(serialized['conv']['weight']['rows'],serialized['conv']['weight']['cols'])
    conv_bias = np.array(serialized['conv']['bias']['b'])
    dense, bias = [],[]
    for d in serialized['dense']:
        dense.append(np.array(d['weight']['w']).reshape(d['weight']['rows'], d['weight']['cols']))
        bias.append(np.array(d['bias']['b']))

    act = soft_relu_np

    X = X @ conv
    
    for i in range(len(X)):
        X[i] += conv_bias
    
    max_tmp = np.abs(X).max()
    if max_tmp > max:
        max = max_tmp
    X = act(X)
    
    iter = 0
    for d,b in zip(dense, bias):
        X = X @ d
        
        for i in range(len(X)):
            X[i] = X[i] + b
        
        max_tmp = np.abs(X).max()
        if max_tmp > max:
            max = max_tmp
        
        if iter != len(dense)-1:
            X = act(X)
        iter += 1
    
    pred = np.argmax(X,axis=1)
    corrects = np.sum(pred == Y.numpy())

    print("Max value in run", max)

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

    elif args.model == "nn50":
        layers = 50

    elif args.model == "nn100":
        layers = 100




    batchsize = 32
    dataHandler = DataHandler(dataset="MNIST", batch_size=batchsize, scale=False)
    corrects = 0

    tot = 0.0
    for X,Y in dataHandler.test_dl:
        corrects += linear_eval(X.double(),Y.double(),serialized)
        tot += batchsize

    print("Accuracy:")
    print(corrects/tot)

