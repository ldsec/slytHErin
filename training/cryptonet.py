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
from packing import *
from os.path import exists

"""
  Implementation of 5-layer CryptoNets on MNIST
"""

## HELPERS FOR JSON SERIALIZATION

#one time thing
def reserialize_cryptonet():
    with open("./models/cryptonet_packed.json", "r") as f:
        data = json.load(f)
        layers = []
        layers.append(Layer(data['conv1']['weight'],data['conv1']['bias']))
        layers.append(Layer(data['pool1']['weight'], data['pool1']['bias']))
        layers.append(Layer(data['pool2']['weight'], data['pool2']['bias']))
        net = Net(layers, 3)
    with open("./models/cryptonet_packed.json", "w") as f:
        json.dump(net.Serialize(),f)


def format_cryptonet(serialized):
    #changes the keys name in the serialized representation
    #the returned dictionary will contain the name of each layers as key,
    #and a dictionary with keys 'weight' and 'bias' and values the row-flattened tensors as value
    #for example, given conv1.weight as key, returns conv1 : {weight: [...], bias: [...]}
    formatted = {}
    ## get all layers name
    layers = [k.split(".")[0] for k in serialized.keys()]
    for l in layers:
        formatted[l] = {}
        l_dict = {}
        for t in types:
            l_dict[t] = serialized[l+"."+t] #e,g conv1.weight
        formatted[l] = l_dict
    return formatted



def extract_param(param_name, param):
    #Params are extracted in row-major order:
    #suppose you have a CONV layer with (k,C,W,H),
    #i.e k kernels wich are tensors of dim C x W x H with C filters of dim W x H
    #each kernel is flattened such that every W x H matrix
    #is flattened by row, for C matrixes. This for every k kernel
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


#Serialize the model. Returns a dictionary which maps layer name to a flattened
#representation of the underlying weight in a json-serializable format
def serialize_cryptonet(model):
    serialized = {}
    for name, p in model.named_parameters():
        serialized[name] = extract_param(name,p)
    
    #change keys s.t each layer is a dict containing a dict with 'weight' and 'bias' as key
    serialized = format_cryptonet(serialized)
    return serialized


def pack_cryptonet(serialized):
    # Packer method for cryptonet
    conv1 = np.array(serialized['conv1']['weight']).reshape(5,1,5,5)
    pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,12,12)
    pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    b1 = np.array(serialized['conv1']['bias'])
    b2 = np.array(serialized['pool1']['bias'])
    b3 = np.array(serialized['pool2']['bias'])

    conv1M,_ = pack_conv(conv1,5,5,2,28,28)
    bias1 = pack_bias(b1, 5, 12*12)
    layer1 = Layer(conv1M, bias1)

    pool1M,_ = pack_pool(pool1)
    bias2 = pack_bias(b2, 100, 1)
    layer2 = Layer(pool1M, bias2)

    pool2M,_ = pack_pool(pool2)
    bias3 = pack_bias(b3, 10, 1)
    layer3 = Layer(pool2M, bias3)

    net = Net([layer1,layer2,layer3], 3)
    return net.Serialize()

class Cryptonet(nn.Module):
  '''
    Simpliefied network used in paper for inference https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/
  '''
  
  def __init__(self, batch_size : int, activation : str, init_method : str, verbose : bool):
    super().__init__()
    self.verbose = verbose
    self.init_method = init_method
    self.batch_size = batch_size

    if activation == "square":
      self.activation = torch.square
    elif activation == "relu":
      self.activation = nn.ReLU()
    elif activation == "relu_approx":
      self.activation = ReLUApprox()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
    self.pool1 = nn.Conv2d(in_channels=5, out_channels=100, kernel_size=12, stride=1000)
    self.pool2 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(100,1), stride=1000)

  def forward(self, x):
    x = self.conv1(x)
    x = self.activation(x)
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

if __name__ == "__main__":
  ##############################
  #                            #
  # TRAINING AND EVAL PIPELINE #
  #                            #
  ##############################
  if exists("./models/cryptonet.pt"):
      print("Packing already trained model...")
      model = torch.load("./models/cryptonet.pt")
      packer = Packer(serialize_cryptonet, pack_cryptonet)
      with open(f'./models/cryptonet_packed.json', 'w') as f:
          json.dump(packer.Pack(model), f)
  else:
    print("Training")
    dataHandler = DataHandler(dataset="MNIST", batch_size=64)

    init = "xavier"
    act = "relu_approx"
    model = Cryptonet(batch_size=dataHandler.batch_size,activation=act,init_method=init,verbose=False).to(device=device)

    logger = Logger("./logs/",f"cryptonet")
    model.apply(model.weights_init)
    train(logger, model, dataHandler, num_epochs=50, lr=5e-4, regularizer='None')
    loss, accuracy = eval(logger, model, dataHandler, loss='MSE')

    score = {"loss":loss, "accuracy":accuracy}

    torch.save(model,'./models/cryptonet.pt')

    packer = Packer(serialize_cryptonet, pack_cryptonet)

    with open(f'./models/cryptonet_packed.json', 'w') as f:
        json.dump(packer.Pack(model), f)
    print("=================================================================")
    print(f"[+] Model with act: {act}, init method: {init}\nAvg test Loss ==> {score['loss']}, Accuracy ==> {score['accuracy']}")


