import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from conv_transform import *
from activation import *
from logger import Logger
from dataHandler import DataHandler
from utils import *

"""
  Implementation of MiniONN on CIFAR10
"""

## HELPERS FOR JSON SERIALIZATION

def format_SimpleNet(serialized):
    ## changes the keys name in the serialized representation (dict)
    formatted = {}
    ## get all layers name
    layers = [k.split(".")[0] for k in serialized.keys()]
    for l in layers:
        formatted[l] = {}
        l_dict = {}
        for t in types:
            l_dict[t] = serialized[l+"."+t]
        formatted[l] = l_dict
    return formatted

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

def serialize_simpleNet(model, format):
    '''
    Serialize the model. Returns a dictionary which maps layer name to a flattened
    representation of the underlying weight in a json-serializable format
    '''
    serialized = {}
    for name, p in model.named_parameters():
        serialized[name] = extract_param(name,p)
    serialized = format(serialized)
    return serialized

def pack_simpleNet(model):
    serialized = serialize_simpleNet(model, format_SimpleNet)
    
    conv1 = np.array(serialized['conv1']['weight']).reshape(5,1,5,5)
    pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,13,13)
    pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    b1 = np.array(serialized['conv1']['bias'])
    b2 = np.array(serialized['pool1']['bias'])
    b3 = np.array(serialized['pool2']['bias'])

    conv1M,_ = pack_conv(conv1,5,2,29)
    bias1 = pack_bias(b1, 5, 13*13)

    pool1M,_ = pack_pool(pool1)
    bias2 = pack_bias(b2, 100, 1)

    pool2M,_ = pack_pool(pool2)
    bias3 = pack_bias(b3, 10, 1)

    packed = {}
    packed['conv1'] = {'weight':conv1M, 'bias': bias1}
    packed['pool1'] = {'weight':pool1M, 'bias': bias2}
    packed['pool2'] = {'weight':pool2M, 'bias': bias3}
    return packed

class MiniONN(nn.Module):
  '''
    CIFAR-10 archtecture from https://eprint.iacr.org/2017/452.pdf
  '''
  
  def __init__(self, init_method : str, verbose : bool):
    super().__init__()
    self.verbose = verbose
    self.init_method = init_method
    self.activation = nn.ReLU()
    self.max = 0.0

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1) #1
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) #3
    self.avg1 = nn.AvgPool2d(kernel_size=2, stride=1) #5
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) #6
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) #8
    self.avg2 = nn.AvgPool2d(kernel_size=2, stride=1) #10
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) #11
    self.conv6 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=2, stride=1) #13
    self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1) #15
    self.dense = nn.Linear(6400,10)

  def forward(self, x):
    x = F.pad(x, [1,1,1,1])
    #print("Input: ", x.shape)
    x = self.conv1(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 1 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    
    x = self.conv2(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 2 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.avg1(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Pool 1 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.conv3(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 3 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.conv4(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 4 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.avg2(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Pool 2 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.conv5(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 5 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.conv6(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 6 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = self.conv7(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    #print("Conv 7 size: ", x.shape)
    x = self.activation(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max

    x = x.reshape(x.shape[0], -1)
    #print("Pre-Dense: ", x.shape)
    x = self.dense(x)
    max = torch.abs(x).max().item()
    if max > self.max:
        self.max = max
    return x
 
  def weights_init(self, m):
    for m in self.children():
      if isinstance(m,nn.Conv2d):
        if self.init_method == "he":
          nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
        elif self.init_method == "xavier":
          nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))
        #elif self.init_method == "uniform":
        #  nn.init.uniform_(m.weight, -0.5, 0.5)
        #elif self.init_method == "norm":
        #  nn.init.normal_(m.weight, 0.0, 1.0)

if __name__ == "__main__":
    dataHandler = DataHandler(dataset="CIFAR", batch_size=32)

    ##############################
    #                            #
    # TRAINING AND EVAL PIPELINE #
    #                            #
    ##############################

   

    model = MiniONN("xavier", verbose=False)
    logger = Logger("./logs/",f"miniONN")
    model.apply(model.weights_init)
    train(logger, model, dataHandler, num_epochs=25, lr=0.001, loss='CSE', optim_algo='Adam', l2_penalty=0.000001, regularizer='L2')
    loss, accuracy = eval(logger, model, dataHandler, loss='CSE')
    print("Max value: ", model.max)  
    ## save
    torch.save(model, f"./models/MiniONN.pt")

    """  
    packed = pack_simpleNet(model)
    with open(f'./models/MiniONN_packed.json', 'w') as f:
      json.dump(packed, f)
    """
    
    
    print("=====================================================================")
    print(f"[+] Avg test Loss ==> {loss}, Accuracy ==> {accuracy}")


