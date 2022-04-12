import torch
import glob
import os
import json
from conv_transform import *
from alexnet import AlexNet
from cryptonet import SimpleNet

os.chdir("./models")

def extract_simpleNet(model):
    serialized = {}
    serialized['conv1'] = {}
    serialized['pool1'] = {}
    serialized['pool2'] = {}
    for p_name, param in model.named_parameters():
        layer_name = p_name.split(".")[0]
        layer_type = p_name.split(".")[1]
        if 'weight' in p_name:
            serialized[layer_name][layer_type] = []
            data = param.data.cpu().numpy()
            if 'conv1' in p_name:
                kernels = data.reshape(5,1,5,5)
                fl_kernels = []
                for i,filter in enumerate(kernels):
                    channels = []
                    for j,channel in enumerate(filter):
                        m = gen_kernel_matrix(channel,5,2,29,tranpose=True)
                        w = []
                        rows = len(m)
                        cols = len(m[0])
                        for r in m:
                            for c in r:    
                                w.append(float(c))
                        channel = {'w':w, 'rows':rows, 'cols':cols}
                        channels.append(channel)
                    fl_kernels.append({'channels':channels})
                serialized[layer_name][layer_type] = fl_kernels
            else:                
                if 'pool1' in p_name:
                    kernels = data.reshape(100,5,13,13)
                    rows = cols = 13
                    fl_kernels = []
                    for i,filter in enumerate(kernels):
                        channels = []
                        for j,channel in enumerate(filter):
                            m = channel
                            w = []
                            rows = len(m)
                            cols = len(m[0])
                            for r in m:
                                for c in r:
                                    w.append(float(c))
                            channel = {'w':w, 'rows':rows, 'cols':cols}
                            channels.append(channel)
                        fl_kernels.append({'channels':channels})
                    serialized[layer_name][layer_type] = fl_kernels
                if 'pool2' in p_name:
                    kernels = data.reshape(10,1,100,1) #(10,100,1,1)--> what should be
                    rows = 1
                    cols = 1
                    fl_kernels = []
                    for i,filter in enumerate(kernels):
                        channels = []
                        m = filter[0] #100x1
                        for j in range(100):
                            w = []
                            w.append(float(m[j][0]))
                            channel = {'w':w, 'rows':rows, 'cols':cols}
                            channels.append(channel)
                        fl_kernels.append({'channels':channels})
                    serialized[layer_name][layer_type] = fl_kernels
        if 'bias' in p_name:
            serialized[layer_name][layer_type] = {}
            data = param.data.cpu().numpy().flatten()
            serialized[layer_name][layer_type]['b']=[]
            serialized[layer_name][layer_type]['len']=len(data)
            for x in data:
                serialized[layer_name][layer_type]['b'].append(x.item())
    
    return serialized

#def extract_alexNet_simplified(model):
#    serialized = {}
#    for p_name, param in model.named_parameters():

if __name__ == '__main__':
    models = [(x,torch.load(x)) for x in glob.iglob("*.pt")]
    for name,m in models:
        if "SimpleNet" in name:
        #    j_name = "simpleNet"
        #    packed = pack_simpleNet(m)
        #    with open(f'{j_name}.json', 'w') as f:
        #        json.dump(packed, f)
            continue
        elif "AlexNet" in name and "simplified" in name:
            j_name = "alexNet_simplified"
            format = format_AlexNet
            #if "simplified" in name:
            #    j_name += "_simplified"
            pack_alexNet(m)