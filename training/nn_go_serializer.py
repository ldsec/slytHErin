import torch
import argparse
import os
from nn import pack_nn
import json
from conv_transform import *
from nn import NN

"""
    Script to serialize nn models from go training in a json format ready to be deserialized in Go implementation 
    Place the json file to be serialized under go training
"""

os.chdir("./models")

## reads models from json format from Go training
def read_nn(json_data, layers=20):
    serialized = {}
    serialized['conv'] = json_data['conv']
    w = np.array(json_data['conv']['weight']['w']).reshape(json_data['conv']['weight']['kernels'],json_data['conv']['weight']['filters'],json_data['conv']['weight']['rows'],json_data['conv']['weight']['cols'])
    serialized['conv']['weight']['w'] = w.tolist()
    for i in range(layers):
        serialized['dense_'+str(i+1)]=json_data["D"]['dense_'+str(i+1)]
        serialized['dense_'+str(i+1)]['weight']['w'] = np.array(serialized['dense_'+str(i+1)]['weight']['w']).reshape(serialized['dense_'+str(i+1)]['weight']['rows'],serialized['dense_'+str(i+1)]['weight']['cols']).tolist()
   
    return serialized

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
        finally:

            j_name = f"{args.model}{args.activation}_packed.json"

            ##go stuff
            with open(f'../go_training/{args.model}{args.activation}_go.json', 'r') as f:
                json_data = json.load(f)
            serialized = read_nn(json_data, layers)
            packed = pack_nn(serialized, layers, transpose_dense=False) #if read_nn set False

            with open(f'{args.model}{args.activation}_packed.json', 'w') as f:
                json.dump(packed, f)
    else:
        print("Model has to be nn")