# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from activation import ReLUApprox, SigmoidApprox
from utils import *
from dataHandler import DataHandlerAlex
from logger import Logger
from conv_transform import *

## JSON SERIALIZATION HELPERS
def extract_param(param_name, param):
    '''
        Params are extracted in row-major order:
        suppose you have a CONV layer with (k,C,W,H), 
        i.e k kernels wich are tensors of dim C x W x H
        each kernel is flattened such that every W x H matrix
        is flattened by row, for C matrixes. This for every k kernel 
    '''
    if 'weight' in param_name:
        weights = []
        data = param.data.cpu().numpy()            
        if 'classifier' in param_name:
            ## for linear layer in AlexNet, transpose first
            data = data.transpose() 
        ## for each kernel filter
        for k in data:
            ## back to 2D
            k = k.flatten()
            for x in k:
                weights.append(x.item()) ##convert to json-serializable
    if 'bias' in param_name:
        weights = []
        data = param.data.cpu().numpy().flatten()
        for k in data:
            weights.append(k.item())
    return weights

def format_AlexNet(serialized):
    formatted = {}
    ## get all layers name
    layers = []
    for k in serialized.keys():
        l = k.split(".")
        if l[0] == 'classifier': ##this is the nn.Sequential layer
            layers.append((".").join(l[0:2]))
        else:
            layers.append(l[0])
    for l in layers:
        formatted[l] = {}
        l_dict = {}
        for t in types:
            l_dict[t] = serialized[l+"."+t]
        formatted[l] = l_dict
    return formatted

def serialize_AlexNet(model, format):
    '''
    Serialize the model. Returns a dictionary which maps layer name to a flattened
    representation of the underlying weight in a json-serializable format
    '''
    serialized = {}
    for name, p in model.named_parameters():
        serialized[name] = extract_param(name,p)
    serialized = format(serialized)
    return serialized

def pack_alexNet(model):
    serialized = serialize_AlexNet(model, format_AlexNet)

    # reshape(chan_out, chan_in, k_size,k_size)
    conv1 = np.array(serialized['conv1']['weight']).reshape(64,3,11,11)
    conv2 = np.array(serialized['conv2']['weight']).reshape(192,64,5,5)
    conv3 = np.array(serialized['conv3']['weight']).reshape(384,192,3,3)
    conv4 = np.array(serialized['conv4']['weight']).reshape(256,284,3,3)
    conv5 = np.array(serialized['conv5']['weight']).reshape(256,256,3,3)
    #pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,13,13)
    #pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    pool1 = np.ones((256,256,3,3))*(1.0/(3**2))
    pool2 = np.ones((256,256,3,3))*(1.0/(3**2))
    dense1 = np.array(serialized['classifier.1']['weight']).reshape(1,1,9216,4096)
    dense2 = np.array(serialized['classifier.4']['weight']).reshape(1,1,4096,4096)
    dense3 = np.array(serialized['classifier.6']['weight']).reshape(1,1,4096,10)


    bias_conv1 = np.array(serialized['conv1']['bias'])
    bias_conv2 = np.array(serialized['conv2']['bias'])
    bias_conv3 = np.array(serialized['conv3']['bias'])
    bias_conv4 = np.array(serialized['conv4']['bias'])
    bias_conv5 = np.array(serialized['conv5']['bias'])
    bias_dense1 = np.array(serialized['classifier.1']['bias'])
    bias_dense2 = np.array(serialized['classifier.4']['bias'])
    bias_dense3 = np.array(serialized['classifier.6']['bias'])

    #linearize layers
    #conv1MD, conv1MT = pack_conv(conv1,11,4,229)
    conv1M = pack_conv_parallel(conv1, 11, 4, 227+2*2)
    bias_conv1M = pack_bias_parallel(bias_conv1, 56*56)

    pool1AM = pack_conv_parallel(pool1,3,2,56)

    conv2M= pack_conv_parallel(conv2,5,1,27+2*2)
    bias_conv2M = pack_bias_parallel(bias_conv2,  27*27)

    pool1BM = pack_conv_parallel(pool1,3,2,27)

    conv3M = pack_conv_parallel(conv3,3,1,13+1*2)
    bias_conv3M = pack_bias_parallel(bias_conv3, 13*13)
    
    conv4M = pack_conv_parallel(conv4,3,1,13+1*2)
    bias_conv4M = pack_bias_parallel(bias_conv4, 13*13)

    conv5M = pack_conv_parallel(conv5,3,1,13+1*2)
    bias_conv5M = pack_bias_parallel(bias_conv5, 13*13)

    pool1CM = pack_conv_parallel(pool1,3,2,13)

    pool2M = pack_conv_parallel(pool2,3,1,6+1*2)

    #compress layers
    P = gen_padding_matrix(25,1,2)
    pool1AM = pad_parallel(P, pool1AM)
    pool1_conv2M, bias_pool1_conv2 = compress_layers(pool1AM, None, conv2M, bias_conv2M)

    P = gen_padding_matrix(13,1,1)
    pool1BM = pad_parallel(P, pool1BM)
    pool1_conv3M, bias_pool1_conv3 = compress_layers(pool1BM, None, conv3M, bias_conv3M)
    
    P = gen_padding_matrix(13,1,1)
    conv4M = pad_parallel(P, conv4M)
    conv5M = pad_parallel(P, conv5M)

    P = gen_padding_matrix(6,1,1)
    pool1CM = pad_parallel(P, pool1CM)
    pool1_pool2M, bias_pool1_pool2 = compress_layers(pool1CM, None , pool2M, None)

    packed = {}
    packed['conv1'] = {'weight':conv1M, 'bias': bias_conv1M}
    packed['conv2'] = {'weight':pool1_conv2M, 'bias': bias_pool1_conv2}
    packed['conv3'] = {'weight':pool1_conv3M, 'bias': bias_pool1_conv3}
    packed['conv4'] = {'weight':conv4M, 'bias': bias_conv4M}
    packed['conv5'] = {'weight':conv5M, 'bias': bias_conv5M}
    packed['pool'] = {'weight':pool1_pool2M, 'bias': bias_pool1_pool2}
    packed['dense1'] = {'weight':pack_linear(dense1), 'bias':{'b': [x.item() for x in bias_dense1], 'len':len(bias_dense1)}}
    packed['dense2'] = {'weight':pack_linear(dense2), 'bias':{'b': [x.item() for x in bias_dense2], 'len':len(bias_dense2)}}
    packed['dense3'] = {'weight':pack_linear(dense3), 'bias':{'b': [x.item() for x in bias_dense3], 'len':len(bias_dense3)}}

    return packed



####################
#                  #
# MODEL DEFINITION #
#                  #
####################

class AlexNet(nn.Module):
  """AlexNet"""
  def __init__(self, simplified : bool, verbose: bool):
    super().__init__()
    self.verbose = verbose
    ## input size for MNIST = 227
    
    self.simplified = simplified
    if self.simplified:
      self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2)
      self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
      self.relu = ReLUApprox()
      self.sigmoid = SigmoidApprox()
    else:
      self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
      self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
      self.relu = nn.ReLU(inplace=True)
      self.sigmoid = nn.Sigmoid()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features= 9216, out_features= 4096),
        self.relu,
        nn.Dropout(p=0.5),
        nn.Linear(in_features= 4096, out_features= 4096),
        self.relu,
        nn.Linear(in_features= 4096, out_features=10),
        self.sigmoid
    )
    ## init weights
    nn.init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv3.weight, a=0, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv4.weight, a=0, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv5.weight, a=0, mode='fan_out', nonlinearity='relu')
    
    i = 0
    for layer in self.classifier.children():
      if isinstance(layer, nn.Linear):
        if i < 2:
          nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
          i += 1
        else:
          nn.init.xavier_uniform_(layer.weight, gain=math.sqrt(2))    
          i += 1
    

  def forward(self,x):
    '''
    Sizes for batch 128:
      torch.Size([128, 3, 227, 227])
      Conv1
      torch.Size([128, 64, 56, 56])
      Pool1
      torch.Size([128, 64, 27, 27]) <<
      Conv2
      torch.Size([128, 192, 27, 27])
      Pool1
      torch.Size([128, 192, 13, 13])
      Conv3
      torch.Size([128, 384, 13, 13])
      Conv4
      torch.Size([128, 256, 13, 13])
      Conv5
      torch.Size([128, 256, 13, 13])
      Pool1
      torch.Size([128, 256, 6, 6])
      Pool2
      torch.Size([128, 256, 6, 6])
      Reshape
      torch.Size([128, 9216])
      Final
      torch.Size([128, 10])
    '''
    print("Start")
    print(x.shape)
    x = self.relu(self.conv1(x))
    print("Conv1")
    print(x.shape)
    x = self.pool1(x)
    print("Pool1")
    print(x.shape)
    x = self.relu(self.conv2(x))
    print("Conv2")
    print(x.shape)
    x = self.pool1(x)
    print("Pool1")
    print(x.shape)
    x = self.relu(self.conv3(x))
    print("Conv3")
    print(x.shape)
    x = self.relu(self.conv4(x))
    print("Conv4")
    print(x.shape)
    x = self.relu(self.conv5(x))
    print("Conv5")
    print(x.shape)
    #print("Conv5",x.max())
    x = self.pool1(x)
    print("Pool1")
    print(x.shape)
    x = self.pool2(x)
    print("Pool2")
    print(x.shape)
    #print("Pools",x.max())
    x = x.reshape(x.shape[0], -1)
    print("Reshape")
    print(x.shape)
    x = self.classifier(x)
    print("Final")
    print(x.shape)
    return x

#########################
#                       #
# TRAIN + TEST PIPELINE #
#                       #
#########################

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
  parser.add_argument("--simplified", help="use HE friendly functions and pooling", action="store_true")
  args = parser.parse_args()
  
  if args.verbose:
    verbose = True
  else:
    verbose = False
  if args.simplified:
    simplified = True
    name = "AlexNet_simplified"
    lr = 0.001
  else:
    simplified = False
    name = "AlexNet"
    lr = 0.05
  
  print(name)
  dataHandler = DataHandlerAlex("MNIST",128)
  logger = Logger("./logs/", name)
  model = AlexNet(simplified=simplified, verbose=verbose).to(device=device)
  train(logger, model, dataHandler, 50, lr=lr, regularizer='None') ##if simplified set lr=0.001
  eval(logger, model, dataHandler, loss='MSE')
  #torch.save(model, f"{name}.pt")