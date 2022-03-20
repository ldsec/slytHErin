from html5lib import serialize
import torch
import glob
import os
from cryptonet import SimpleNet
from alexnet import AlexNet
from torch.nn.parameter import Parameter
import json
import time

os.chdir("./models")

types = ['weight', 'bias']

def format_SimpleNet(serialized):
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

def format_AlexNet(serialized):
    formatted = {}
    ## get all layers name
    layers = []
    for k in serialized.keys():
        l = k.split(".")
        if l [0] == 'classifier': ##this is the nn.Sequential layer
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
        
def extract_param(param_name, param):
    """
        Params are extracted in row-major order:
        suppose you have a CONV layer with (k,C,W,H), 
        i.e k kernels wich are tensors of dim C x W x H
        each kernel is flattened such that every W x H matrix
        is flattened by row, for C matrixes. This for every k kernel 
    """
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
    if 'bias' in p_name:
        weights = []
        data = param.data.cpu().numpy()
        for k in data:
            weights.append(k.item())
    return weights


models = [(x,torch.load(x)) for x in glob.iglob("*.pt")]
for name,m in models:
    if "SimpleNet" in name:
        j_name = "simpleNet"
        format = format_SimpleNet
    elif "AlexNet" in name:
        j_name = "alexNet"
        format = format_AlexNet
        if "simplified" in name:
            j_name += "_simplified"
    print(f"Serializing {j_name}...")
    serialized = {}
    for p_name, param in m.named_parameters():
        serialized[p_name] = extract_param(p_name, param)
    serialized = format(serialized)
    with open(f'{j_name}.json', 'w') as f:
        json.dump(serialized, f)
    print()
    print("Done!")