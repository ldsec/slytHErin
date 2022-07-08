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
  Implementation of 5-layer CryptoNets on MNIST
"""

## HELPERS FOR JSON SERIALIZATION

def format_cryptonet(serialized):
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

def serialize_cryptonet(model, format):
    '''
    Serialize the model. Returns a dictionary which maps layer name to a flattened
    representation of the underlying weight in a json-serializable format
    '''
    serialized = {}
    for name, p in model.named_parameters():
        serialized[name] = extract_param(name,p)
    serialized = format(serialized)
    return serialized

def pack_cryptonet(model):
    serialized = serialize_cryptonet(model, format_cryptonet)
    
    conv1 = np.array(serialized['conv1']['weight']).reshape(5,1,5,5)
    pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,12,12)
    pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    b1 = np.array(serialized['conv1']['bias'])
    b2 = np.array(serialized['pool1']['bias'])
    b3 = np.array(serialized['pool2']['bias'])

    conv1M,_ = pack_conv(conv1,5,2,28)
    bias1 = pack_bias(b1, 5, 12*12)

    pool1M,_ = pack_pool(pool1)
    bias2 = pack_bias(b2, 100, 1)

    pool2M,_ = pack_pool(pool2)
    bias3 = pack_bias(b3, 10, 1)

    packed = {}
    packed['conv1'] = {'weight':conv1M, 'bias': bias1}
    packed['pool1'] = {'weight':pool1M, 'bias': bias2}
    packed['pool2'] = {'weight':pool2M, 'bias': bias3}
    return packed

class cryptonet(nn.Module):
  '''
    Simpliefied network used in paper for inference https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/
  '''
  
  def __init__(self, batch_size : int, activation : str, sigmoid : str, init_method : str, verbose : bool):
    super().__init__()
    self.verbose = verbose
    self.init_method = init_method
    self.batch_size = batch_size
    self.activation = activation
    self.sigmoid = sigmoid

    if activation == "square":
      self.activation = torch.square
    elif activation == "relu":
      self.activation = nn.ReLU()
    elif activation == "relu_approx":
      self.activation = ReLUApprox()

    if sigmoid == "sigmoid":
      self.sigmoid = nn.Sigmoid()
    elif sigmoid == "approx":
      self.sigmoid = SigmoidApprox()
    elif sigmoid == "none":
      self.sigmoid = identity

    self.pad = F.pad
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
    self.pool1 = nn.Conv2d(in_channels=5, out_channels=100, kernel_size=12, stride=1000)
    self.pool2 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(100,1), stride=1000)

  def forward(self, x):
    #x = self.pad(x, (1,1,1,1))
    x = self.conv1(x)
    print(x.shape)
    x = self.activation(x)
    #print(x.shape)
    x = self.pool1(x)
    x = x.reshape([self.batch_size,1,100,1]) #batch_size tensors in 1 channel, 100x1
    x = self.activation(x)
    x = self.pool2(x)
    x = x.reshape(x.shape[0], -1)
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
    dataHandler = DataHandler(dataset="MNIST", batch_size=16)

    ##############################
    #                            #
    # TRAINING AND EVAL PIPELINE #
    #                            #
    ##############################

    ## TEST
    ## init models 
    #methods = ["he", "xavier", "random"] ##he init blows up values with square
    #methods = ["xavier","he"]
    #activations = ["relu_approx","relu"]
    #models = {}
    #sigmoid = True
    #for method in methods:
    #  for activation in activations:
    #    models[method+"_"+activation] = cryptonet(batch_size=dataHandler.batch_size,
    #                                    activation=activation,
    #                                    init_method=method,
    #                                    verbose=False,
    #                                    sigmoid=sigmoid).to(device=device)
    ## TEST

    models = {}
    #models["xavier_relu"] = cryptonet(batch_size=dataHandler.batch_size,
    #                                    activation="relu",
    #                                    init_method="xavier",
    #                                    verbose=False,
    #                                    sigmoid=True).to(device=device)
    #models["he_relu"] = cryptonet(batch_size=dataHandler.batch_size,
    #                                    activation="relu",
    #                                    init_method="he",
    #                                    verbose=False,
    #                                    sigmoid=True).to(device=device)

    ## Most promising model. With approximated sigmoid we can increase accuracy
    ## up to 96%, but it's not faithful to the original model, plus it is more complex
    models["xavier_relu_approx"] = cryptonet(batch_size=dataHandler.batch_size,
                                        activation="relu_approx",
                                        init_method="xavier",
                                        sigmoid="none",
                                        verbose=False).to(device=device)

    #models["he_relu_approx"] = cryptonet(batch_size=dataHandler.batch_size,
    #                                    activation="relu_approx",
    #                                    init_method="he",
    #                                    verbose=False,
    #                                    sigmoid=False).to(device=device)

    scores = {}

    for key, model in models.items():
      logger = Logger("./logs/",f"cryptonet_{key}")
      model.apply(model.weights_init)
      train(logger, model, dataHandler, num_epochs=50, lr=5e-4, regularizer='None')
      loss, accuracy = eval(logger, model, dataHandler, loss='MSE')
      
      scores[key] = {"loss":loss, "accuracy":accuracy}
      
      ## save
      torch.save(model, f"./models/cryptonet_{key}.pt")
      packed = pack_cryptonet(model)
      with open(f'./models/cryptonet_{key}_packed.json', 'w') as f:
        json.dump(packed, f)

    
    
    for key, metrics in scores.items():
      print("=====================================================================")
      print(f"[+] Model with {key}: Avg test Loss ==> {metrics['loss']}, Accuracy ==> {metrics['accuracy']}")


