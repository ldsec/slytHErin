import torch
import argparse
import os
import glob
import json
from conv_transform import *
from nn import NN
from cryptonet import SimpleNet

"""
    Script to serialize models in a json format ready to be deserialized in Go implementation 
"""


os.chdir("./models")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="simplenet, nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    
    if args.model == "simplenet":
        #models = [(x,torch.load("")) for x in glob.iglob("*.pt")]
        models = [(x,torch.load("")) for x in glob.iglob("SimpleNet*.pt")]
        for name,m in models:
            if "SimpleNet" in name:
                j_name = "simpleNet"
                packed = pack_simpleNet(m)
                with open(f'{j_name}.json', 'w') as f:
                    json.dump(packed, f)

            ##alexnet is currently too bit to be handled
            #elif "AlexNet" in name and "simplified" in name:
            #    j_name = "alexNet_simplified"
            #    format = format_AlexNet
            #    #if "simplified" in name:
            #    #    j_name += "_simplified"
            #    pack_alexNet(m)

    elif "nn" in args.model:
        #with open(f'{args.model}.json', 'r') as f:
        #    serialized = json.load(f)
        if args.model == "nn20":
            model = torch.load("nn20.pt")
            layers = 20
        elif args.model == "nn50":
            model = torch.load("nn50.pt")
            layers = 50
        elif args.model == "nn100":
            model = torch.load("nn100.pt")
            layers = 100
        j_name = f"nn{layers}_packed.json"
        
        ##go stuff
        with open(f'nn_{layers}.json', 'r') as f:
            json_data = json.load(f)
        serialized = read_nn(json_data, layers)
        
        #conv,bias_conv,dense,bias = extract_nn(model)
        #serialized = serialize_nn(conv,bias_conv,dense,bias,layers+1)
        
        packed = pack_nn(serialized, layers, transpose_dense=False) #if read_nn set False
        with open(j_name, 'w') as f:
            json.dump(packed, f)
    else:
        exit("Define model")