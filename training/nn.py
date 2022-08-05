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
import torch
import argparse
import os
import json
from conv_transform import *
from packing import *
from os.path import exists

"""
    Script to serialize nn models from go training in a json format ready to be deserialized in Go implementation 
    Place the json file to be serialized under ./go_training
"""

os.chdir("./models")

def resereliaze_nn(path):
    with open(path, "r") as f:
        data = json.load(f)
        layers = []
        layers.append(Layer(data['conv']['weight'], data['conv']['bias']))
        for i in range(data['layers']):
            layers.append(Layer(data['dense'][i]['weight'],data['dense'][i]['bias']))
        net = Net(layers, data['layers'])
    with open(path, "w") as f:
        json.dump(net.Serialize(),f)

## serialize models from json format from Go training
def serialize_nn(json_data):
    layers = len(json_data['D'])
    serialized = {}
    serialized['conv'] = json_data['conv']
    w = np.array(json_data['conv']['weight']['w']).reshape(json_data['conv']['weight']['kernels'],json_data['conv']['weight']['filters'],json_data['conv']['weight']['rows'],json_data['conv']['weight']['cols'])
    serialized['conv']['weight']['w'] = w.tolist()
    for i in range(layers):
        serialized['dense_'+str(i+1)]=json_data["D"]['dense_'+str(i+1)]
        serialized['dense_'+str(i+1)]['weight']['w'] = np.array(serialized['dense_'+str(i+1)]['weight']['w']).reshape(serialized['dense_'+str(i+1)]['weight']['rows'],serialized['dense_'+str(i+1)]['weight']['cols']).tolist()
   
    serialized['numLayers'] = layers    
    return serialized

## given a serialized representation from read_nn, packs it in json format for Go inference under HE
def pack_nn(serialized):
    num_layers = serialized['numLayers']
    num_chans = serialized['conv']['weight']['kernels']
    conv_matrix,_ = pack_conv(np.array(serialized['conv']['weight']['w']),
        serialized['conv']['weight']['rows'],
        serialized['conv']['weight']['cols'],
        1,
        28,28)

   
    conv_bias = pack_bias(np.array(serialized['conv']['bias']['b']), num_chans, conv_matrix['cols']//num_chans)
    layers = [Layer(conv_matrix, conv_bias)]
    
    for i in range(num_layers):
        w = np.array(serialized['dense_'+str(i+1)]['weight']['w'])
        layers.append(Layer(
            pack_linear(w),
            pack_bias(serialized['dense_'+str(i+1)]['bias']['b'],
                      serialized['dense_'+str(i+1)]['bias']['cols'],1)))
    
    net = Net(layers, num_layers)
    return net.Serialize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="nn20, nn50, nn100",type=str)
    parser.add_argument("--activation", help="poly", nargs='?', const="", type=str)
    args = parser.parse_args()
    if "nn" in args.model:
        if args.model == "nn20":
            layers = 20
        elif args.model == "nn50":
            layers = 50
        elif args.model == "nn100":
            layers = 100

        try:
            if "poly" in args.activation:
                args.activation = "_" + args.activation
        except:
            args.activation = ""
        finally:

            j_name = f"{args.model}{args.activation}_packed.json"

            #if exists(j_name):
            #    resereliaze_nn(j_name)
            #    exit(0)

            ##go stuff
            with open(f'../nn_go_training/{args.model}{args.activation}_go.json', 'r') as f:
                json_data = json.load(f)
                packer = Packer(serialize_nn, pack_nn)
                packed = packer.Pack(json_data)

            with open(f'{args.model}{args.activation}_packed.json', 'w') as f:
                json.dump(packed, f)
    else:
        print("Model has to be nn")