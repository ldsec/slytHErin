from re import A
import numpy as np
import torch
import json
import math
from collections import deque
from activation import relu_approx
from dataHandler import DataHandler
from cryptonet import SimpleNet
## JP function
def gen_kernel_matrix_JP(kernel, stride_h, stride_v, dim):
    m = [[0 for i in range(dim*dim)] for j in range(dim*dim)]

    kernel_flattened = [0 for i in range(len(kernel) * dim - (dim-len(kernel)))]

    for i in range(len(kernel)):
        for j in range(len(kernel)):
            kernel_flattened[i*dim + j] = kernel[i][j]
    a = (dim - len(kernel) + stride_v)//stride_v
    b = (dim - len(kernel) + stride_h)//stride_h

    for i in range(a):
        for j in range(b):
            idx_j = j * stride_h + i * stride_v * dim
            for k in range(len(kernel_flattened)):
                m[i*b+j][idx_j+k] = kernel_flattened[k]
    #for i in m:
    #    print(i)
    return m

def rotR(a,k):
    rot_a = deque(a)
    rot_a.rotate(k)
    return list(rot_a)

def gen_kernel_matrix(kernel, kernel_size, stride, dim, tranpose=False):
    """
    Transform each kernel of a conv layer in a matrix m:
        Conv(input,k) --> m @ input.flatten()
        where @ is the usual matrix multiplication

    """
    out_dim = math.floor((dim - kernel_size)/stride + 1)
    ## create a matrix of dimention nxm, where:
    ##  n is the lenght of flattened output
    ##  m is the dim of flattened input
    m = [[0 for i in range(dim*dim)] for j in range(out_dim*out_dim)]
    kernel_flattened = m[0] ## 0 vector of dim^2
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            ## fill the vector by assigning each kernel weight to
            ## its position in the input matrix in the first patch
            ## we will then rotate this
            kernel_flattened[i*dim + j] = kernel[i][j]

    patch_idx = len(kernel)-1
    init_patch_idx = patch_idx ## horizontal index
    rotations = 0
    row = 0 #which row in input
    for i in range(len(m)):
        v = rotR(kernel_flattened, rotations)
        m[i] = v
        patch_idx = stride + patch_idx ##simulate the patch movement during conv
        rotations = rotations + stride
        if patch_idx >= dim:
            patch_idx = init_patch_idx
            rotations = 0
            rotations = row*dim + stride*dim ## the row we are now + skip stride (vertical) rows
            row = row + stride # skipped stride rows
    #for r in m:
    #    print(r)
    if tranpose:
        rows = len(m)
        cols = len(m[0])
        m_t = [[0 for i in range(rows)] for j in range(cols)]
        for j in range(cols):
            for i in range(rows):
                m_t[j][i] = m[i][j]
        m = m_t

    return m


"""
    This script contains test evaluated with the original model in pytorch
    of different linear approximations of the convolutional layers
    to be evalauted in the inference directory, in Go, under
    homomorphic encryption

    Python allows developing quick PoC for various methods, and also allows
    quick packaging of matrix with numpy agility

    The idea:
        gen_kernel_matrix(k) returns for each channel of a kernel, a matrix m s.t
        m @ x.T = conv(k,x)

        if we have n kernels , we can generate a matrix M

        M = | m_1 |
            | m_2 |
            | ... |
            | m_n |

        s.t 

"""
if __name__ == "__main__":
    ### TEST 1 --> simpleNet simple convolution 
    #with open("./models/simpleNet.json") as f:
    #    model = json.loads(f.read())
    #    k = np.array(model['conv1']['weight']).reshape(5,5,5)
    #    M = []
    #    M_JP = []
    #    for kernel in k:
    #        M_JP.append(gen_kernel_matrix_JP(kernel,2,2,29))
    #        M.append(gen_kernel_matrix(kernel,5,2,29,True))
#
    #    data = np.random.rand(2,29,29)
#
    #    data_t = torch.from_numpy(data).reshape(2,1,29,29)
    #    k_t = torch.from_numpy(k).reshape(5,1,5,5)
    #    c = torch.nn.functional.conv2d(data_t,k_t,stride=2, groups=1).reshape(2,5,13,13)
    #    c_1 = torch.nn.functional.conv2d(data_t[0].reshape(1,1,29,29),k_t,stride=2, groups=1).reshape(1,5,13,13)
    #    c_2 = torch.nn.functional.conv2d(data_t[1].reshape(1,1,29,29),k_t,stride=2, groups=1).reshape(1,5,13,13)
    #    loss = []
    #    data = data.reshape(2,841)
    #    for j in range(2):
    #        for i,m in enumerate(M):
    #            a = c[j][i].flatten()
    #            a = a.flatten()
    #            b = data @ m
    #            #print("Diff between conv and transform_conv:")
    #            dist = np.linalg.norm(a.numpy()-b[j].flatten())
    #            #print("conv:",a.numpy())
    #            #print("linear:",b)
    #            loss.append(dist)
    #    print("Avg euclidean distance between conv and transform_conv:")
    #    print(abs(np.average(np.array(loss))))
#

    ## TEST 2 --> alexNet RGB conv kind of
    #stride=2
    #kernel_size=5
    #input_size=29
    #in_chans = 3
    #out_chans = 64
    #output_size = math.floor((input_size - kernel_size)/stride + 1)
    #k = np.random.rand(out_chans,in_chans,kernel_size,kernel_size)
    #flattened_kernels = []
    #for kernel in k:
    #    ## each kernel has 3 filter, 1 per channel
    #    #print(kernel.shape)
    #    flattened_kernel = []
    #    for channel in kernel:
    #        m = gen_kernel_matrix(channel,kernel_size=kernel_size,stride=stride,dim=input_size)
    #        flattened_kernel.append(m)
    #    flattened_kernels.append(flattened_kernel)
    #
    #data = np.random.rand(in_chans,input_size,input_size)
    #data_t = torch.from_numpy(data).reshape(1,in_chans,input_size,input_size)
    #k_t = torch.from_numpy(k).reshape(out_chans,in_chans,kernel_size,kernel_size)
    #c = torch.nn.functional.conv2d(data_t,k_t,stride=stride).reshape(out_chans,output_size,output_size)
    #loss = []
    #for i,fl_k in enumerate(flattened_kernels):
    #    a = c[i].flatten()
    #    b = np.zeros((1,output_size*output_size))
    #    for channel,d in zip(fl_k,data):
    #        ## compute the linearized conv for each channel and sum'em up
    #        #print(d.shape)
    #        #print(f.shape)
    #        b = b + channel @ d.flatten()
    #    dist = np.linalg.norm(a.numpy()-b)
    #    #print("conv:",a.numpy())
    #    #print("linear:",b)
    #    loss.append(dist)
    #print("Avg euclidean distance between conv and transform_conv:")
    #print(abs(np.average(np.array(loss))))
#
    ## TEST 3 --> SimpleNet pooling
    ###btw pooling is for free, it is simply element wise multiplication plus sum of all elements, so log(n) rots
    #with open("./models/simpleNet.json") as f:
    #    batch = 2
    #    model = json.loads(f.read())
    #    k = np.array(model['pool2']['weight']).reshape(10,1,100,1)
    #    #k = np.zeros((4,1,2,1))
    #    #for i in range(2):
    #    #    for j in range(2):
    #    #        k[i][0][j] = 2*i+2*j
    #    k_t = torch.from_numpy(k)
    #    data = np.random.rand(batch,1,100,1)
    #    #data = np.zeros((2,1,2,1))
    #    #for i in range(2):
    #    #    for j in range(2):
    #    #        data[i][0][j] = i+j
    #    c = torch.nn.functional.conv2d(torch.from_numpy(data),k_t,stride=1)
    #    print("data",data)
    #    print("k",k)
    #    print(c)
    #    output = np.zeros((10,batch))
    #    for i,kernel in enumerate(k):
    #        res_kernel = np.zeros((batch,1))
    #        for j,channel in enumerate(kernel[0]):
    #            d = np.zeros((batch,1))
    #            for z in range(batch):
    #                d[z] = data[z][0][j]
    #            vector_channel = np.ones((batch,1))*channel
    #            #print("channel", vector_channel)
    #            res_kernel = res_kernel + vector_channel * d
    #        output[i] = res_kernel.T
    #    print(c.reshape(batch,10)-output.T)
    #   
    ##TEST 4 -> simplenet pipeline
    with open("./models/simpleNet.json") as f:
        device = torch.device('cpu')
        #torch_model = SimpleNet(64, 'relu_approx', 'none', 'xavier', False)
        #torch_model.load_state_dict(torch.load("./models/SimpleNet_xavier_relu_approx.pt",map_location=device))
        torch_model = torch.load("./models/SimpleNet_xavier_relu_approx.pt",map_location=device).double()
        torch_model.batch_size = 64
        torch_model.eval()
        
        model = json.loads(f.read())
        conv1 = np.array(model['conv1']['weight']).reshape(5,1,5,5)
        pool1 = np.array(model['pool1']['weight']).reshape(100,5,13,13)
        pool2 = np.array(model['pool2']['weight']).reshape(10,1,100,1)
        b1 = np.array(model['conv1']['bias'])
        b2 = np.array(model['pool1']['bias'])
        b3 = np.array(model['pool2']['bias'])
        dh = DataHandler("MNIST", 64)

        ## pack kernel matrixes
        flattened_kernels = []
        for kernel in conv1:
            ## each kernel has 3 filter, 1 per channel
            #print(kernel.shape)
            channel = kernel[0]
            m = gen_kernel_matrix(channel,kernel_size=5,stride=2,dim=29)
            flattened_kernels.append(np.array(m))
        conv1M = np.vstack(flattened_kernels)
        #print(conv1M.shape)

        ## block matrix of 100 channels stacked vertically, stacked horizontaly
        kernel_matrices = []
        for kernel in pool1:
            channels = []
            for channel in kernel:
                channels.append(channel.flatten().T)
            channel_matrix = np.vstack(channels).flatten().T
            kernel_matrices.append(channel_matrix)
        pool1M = np.vstack(kernel_matrices)
        #print(pool1M.shape)

        kernel_matrices = []
        for kernel in pool2:
            channels = []
            for channel in kernel:
                channels.append(channel.flatten().T)
            channel_matrix = np.vstack(channels).flatten().T
            kernel_matrices.append(channel_matrix)
        pool2M = np.vstack(kernel_matrices)
        #print(pool2M.shape)

        correct_conv = 0
        correct_linear = 0
        correct_pytorch = 0
        tot = 0
        for X,Y in dh.test_dl:
            X = X.double()
            X_torch = X
            X = torch.nn.functional.pad(X,(1,0,1,0))
            X_t = X
            X = X_t.numpy()
            X_flat = np.zeros((64,841))
            for i in range(64):
                X_flat[i] = X[i][0].flatten()
            c1 = torch.nn.functional.conv2d(X_t,torch.from_numpy(conv1),bias=torch.from_numpy(b1),stride=2)
            d1 = X_flat @ conv1M.T
            
            bias = np.ones(d1.shape)
            for i in range(d1.shape[0]):
                for j in range(d1.shape[1]):
                    idx = j // 169
                    d1[i][j] = d1[i][j] + b1[idx] 
            
            c1_flat = np.zeros(d1.shape)
            for i,c in enumerate(c1):
                r = []
                for ch in c:
                    r.append(ch.flatten())
                c1_flat[i] = np.hstack(r)
            dist = np.linalg.norm(c1_flat-d1)
            print("conv1", dist)

            c2 = torch.nn.functional.conv2d(c1, torch.from_numpy(pool1),bias=torch.from_numpy(b2),stride=1000)
            c2 = relu_approx(c2)
            c2_f = c2.reshape(c2.shape[0],-1)
            d2 = d1 @ pool1M.T
            bias = np.ones(d2.shape)
            
            for j in range(d2.shape[1]):
                for i in range(d2.shape[0]):
                    bias[i][j] = bias[i][j]*b2[j]
            d2 = relu_approx(torch.from_numpy(d2+bias)).numpy()
            dist = np.linalg.norm(c2_f.numpy()-d2)
            print("pool1", dist)

            c3 = torch.nn.functional.conv2d(c2.reshape(64,1,100,1), torch.from_numpy(pool2),bias=torch.from_numpy(b3), stride=1000)
            c3 = relu_approx(c3)
            c3 = c3.reshape(c3.shape[0],-1)
            d3 = d2 @ pool2M.T
            bias = np.ones(d3.shape)
            for j in range(d3.shape[1]):
                for i in range(d3.shape[0]):
                    bias[i][j] = bias[i][j]*b3[j]
            d3 = relu_approx(torch.from_numpy(d3+bias)).numpy()
            dist = np.linalg.norm(c3.numpy()-d3)
            print("pool2", dist)
            #print(d3.shape)

            _,pred = torch_model(X_torch.double()).max(1)
            _,pred_c = c3.max(1)
            pred_l = np.argmax(d3,axis=1)
            correct_pytorch = correct_pytorch + (pred == Y).sum().item()
            correct_conv = correct_conv + (pred_c == Y).sum().item()
            correct_linear = correct_linear + np.sum(pred_l == Y.numpy())
            tot = tot + Y.shape[0]

        print(f"Accuracy torch {correct_pytorch/tot}")
        print(f"Accuracy conv {correct_conv/tot}")
        print(f"Accuracy linear {correct_linear/tot}")

