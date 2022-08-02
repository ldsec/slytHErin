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
    Script for testing nn with approximations for homomorphic encryption
"""
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


def linear_eval(X, Y, serialized, activation):
    """
        Evaluates the HE friendly pipeline
    """

    X = X.reshape(X.shape[0], -1)
    X = X.numpy()

    conv = np.array(serialized['layers'][0]['weight']['w']).reshape(serialized['layers'][0]['weight']['rows'],
                                                               serialized['layers'][0]['weight']['cols'])
    conv_bias = np.array(serialized['layers'][0]['bias']['b'])
    dense, bias = [], []

    for d in serialized['layers'][1:]:
        dense.append(np.array(d['weight']['w']).reshape(d['weight']['rows'], d['weight']['cols']))
        bias.append(np.array(d['bias']['b']))

    act = activation

    X = X @ conv

    for i in range(len(X)):
        X[i] += conv_bias

    max_tmp = X.flatten()[np.argmax(X)]
    min_tmp = X.flatten()[np.argmin(X)]

    if max_tmp > intervals[0][1]:
        intervals[0][1] = max_tmp
    if min_tmp < intervals[0][0]:
        intervals[0][0] = min_tmp
    X = act(X)

    iter = 0
    for d, b in zip(dense, bias):
        X = X @ d

        for i in range(len(X)):
            X[i] = X[i] + b

        if iter != len(dense) - 1:
            max_tmp = X.flatten()[np.argmax(X)]
            min_tmp = X.flatten()[np.argmin(X)]

            if max_tmp > intervals[iter + 1][1]:
                intervals[iter+1][1] = max_tmp
            if min_tmp < intervals[iter + 1][0]:
                intervals[iter+1][0] = min_tmp
            X = act(X)

        iter += 1

    pred = np.argmax(X, axis=1)
    corrects = np.sum(pred == Y.numpy())

    return corrects

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="nn20, nn50, nn100",type=str)
    parser.add_argument("--activation", help="poly", nargs='?', const="", type=str)
    args = parser.parse_args()
    device = 'cpu'
    if "nn" not in args.model:
        print("call --model nn20, nn50 or nn100")
        exit()
    try:
        if "poly" in args.activation:
            args.activation = "_"+args.activation
            activation = silu_np
        else:
            activation = soft_relu_np
    except:
        args.activation = ""
        activation = soft_relu_np

    print(f"Model: {args.model}{args.activation}")
    with open(f'./models/{args.model}{args.activation}_packed.json', 'r') as f:
        serialized = json.load(f)
    if args.model == "nn20":
        layers = 20

    elif args.model == "nn50":
        layers = 50

    elif args.model == "nn100":
        layers = 100

    intervals = []
    for i in range(layers):
        intervals.append([0, 0])


    batchsize = 512
    dataHandler = DataHandler(dataset="MNIST", batch_size=batchsize, scale=False)
    corrects = 0

    tot = 0.0
    for X,Y in dataHandler.test_dl:
        corrects += linear_eval(X.double(),Y.double(),serialized, activation)
        tot += batchsize

    print("Accuracy:")
    print(corrects/tot)
    print("Intervals")
    for i, interval in enumerate(intervals):
        print("Layer: ", i + 1)
        print("Min: ", interval[0])
        print("Max: ", interval[1])
        print()

    ## dump intervals
    intervals_json = {'intervals':[{'a': x[0], 'b':x[1]} for x in intervals]}
    with open(f'./models/{args.model}{args.activation}_intervals.json', 'w') as f:
        json.dump(intervals_json, f)



