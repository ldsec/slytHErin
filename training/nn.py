import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import numpy as np
from utils import *
from activation import *
from logger import Logger
from dataHandler import *
from conv_transform import *
from activation import *

"""
    Train NN
    run via python3 nn.py --model nnX for X = 20,50,100

    Journal:
        -   Best is model with ReLU, SiLU also performs good
        -   These models create intermediate values outside approx range
        -   Training with approximated funcs does not work
"""
## HELPERS FOR JSON SERIALIZATION


def extract_nn(model):
    ## takes model from pytorch and returns numpy representations of the layers
    dense = []
    bias = []
    for name, param in model.named_parameters():
        if 'conv' in name:
            if 'weight' in name:
                conv = param.data.cpu().numpy()
            else:
                bias_conv = param.data.cpu().numpy()
        else:
            if 'weight' in name:
                dense.append(param.data.cpu().numpy())
            else:
                bias.append(param.data.cpu().numpy())
    return conv, bias_conv, dense, bias

 
def serialize_nn(conv, bias_conv, dense, bias, layers):
    ## returns a dictionary representation from a model extracted with extract_nn
    layers = layers+1
    serialized = {}
    serialized['conv'] = {}
    serialized['conv']['weight'] = {'w': conv.tolist(),
    'kernels': conv.shape[0],
    'filters': conv.shape[1],
    'rows': conv.shape[2],
    'cols': conv.shape[3]}
    serialized['conv']['bias'] = {'b': bias_conv.tolist(), 'rows': 1, 'cols': bias_conv.shape[0]}
    for i in range(layers-1):
        serialized['dense_'+str(i+1)] = {}
        serialized['dense_'+str(i+1)]['weight'] = {'w': dense[i].tolist(), 'rows': dense[i].shape[0], 'cols': dense[i].shape[1]}
        serialized['dense_'+str(i+1)]['bias'] = {'b': bias[i].tolist(), 'rows': 1, 'cols': bias[i].shape[0]}
    return serialized

def pack_nn(serialized, layers, transpose_dense=True):
    ## given a serialized representation, packs it in json format for Go inference under HE 
    packed = {}
    num_chans = serialized['conv']['weight']['kernels']
    conv_matrix,_ = pack_conv_rect(np.array(serialized['conv']['weight']['w']),
        serialized['conv']['weight']['rows'],
        serialized['conv']['weight']['cols'],
        1,
        28,28)

    packed['conv'] = {
        'weight': conv_matrix,
        'bias': pack_bias(np.array(serialized['conv']['bias']['b']), num_chans, conv_matrix['cols']//num_chans)}
    packed['dense'] = []
    for i in range(layers):
        w = np.array(serialized['dense_'+str(i+1)]['weight']['w'])
        if transpose_dense:
            w = w.T
        packed['dense'].append({
            'weight': pack_linear(w),
            'bias': {'b':serialized['dense_'+str(i+1)]['bias']['b'], 'len':serialized['dense_'+str(i+1)]['bias']['cols']}})
    
    packed['layers'] = layers
    return packed

## MODEL

class NN(nn.Module):
  '''
    Retraining of ZAMA NN architecture 
  '''
  
  def __init__(self, layers, verbose = False):
    super().__init__()
    self.verbose = verbose
    self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(10,11), stride=1)
    dense = []
    dense.append(nn.Linear(840,92,True))
    for _ in range(layers-2):
        dense.append(nn.Linear(92,92,True))
    dense.append(nn.Linear(92,10,True))
    self.dense = nn.ModuleList(dense)

    for layer in self.dense:
        assert(layer.weight.requires_grad == True)

    self.max = 0 #records max interval of x
    
    self.activation = nn.ReLU() ##lr = 0.001, Adam, CSE --> max value recorded is 1080
    #self.activation = nn.Sigmoid()
    #self.activation = ReLUApprox()
    #self.activation = SigmoidApprox()
    #self.activation = SILUApprox()
    #self.activation = nn.SiLU() ##lr=0.003 --> max value 389
    #self.activation = nn.Softplus(threshold=30)

  def forward(self, x):
    ## preprocessing
    x = self.conv(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    x = self.activation(x)
    if max > self.max:
        self.max = max
    x = x.reshape(x.shape[0],-1) #flatten
    
    for i,layer in enumerate(self.dense):
        x = layer(x)
        max = torch.abs(x).max().item()
        if max > self.max:
            self.max = max
        if i != len(self.dense)-1:
            x = self.activation(x)
            max = torch.abs(x).max().item()
            if max > self.max:
                self.max = max
    return x
 
  def weights_init(self, m):
    for m in self.children():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
        
def train_nn():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="simplenet, nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    
    if args.model == "nn20":
        layers = 20
    elif args.model == "nn50":
        layers = 50
    elif args.model == "nn100":
        layers = 100

    dataHandler = DataHandler(dataset="MNIST", batch_size=32)
    model = NN(layers, verbose=False)
    logger = Logger("./logs/",f"nn{layers}")
    model.apply(model.weights_init)
    start = time.time()
    train(logger, model, dataHandler, num_epochs=20, lr=1e-3, momentum=0.9, l1l2_penalty=1e-5, optim_algo='Adam', loss='CSE', regularizer='Elastic')
    end = time.time()
    print("--- %s seconds for train---" % (end - start))
    print(f"Max value recorded in training: {model.max}")
    loss, accuracy = eval(logger, model, dataHandler, loss='CSE')
    print(f"Loss test: {loss}")
    print(f"Accuracy test: {accuracy}")

    torch.save(model, f"./models/nn{layers}.pt")
    
    conv, bias_conv, dense, bias = extract_nn(model)
    
    serialized = serialize_nn(conv,bias_conv,dense,bias,layers)
    packed = pack_nn(serialized,layers)
    
    with open(f'./models/{args.model}_packed.json', 'w') as f:
        json.dump(packed, f)
    
if __name__ == '__main__':
    ## use for training a new model
    train_nn()