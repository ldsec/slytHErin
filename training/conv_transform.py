import numpy as np
import torch
import json
import math
from collections import deque
'''
## JP function
def gen_kernel_matrix(kernel, stride_h, stride_v, dim):
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
'''
def rotR(a,k):
    rot_a = deque(a)
    rot_a.rotate(k)
    return list(rot_a)

def gen_kernel_matrix(kernel, kernel_size, stride, dim):
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
    for i in range(len(m)):
        v = rotR(kernel_flattened, rotations)
        m[i] = v
        #print("row",i)
        #print("rots", rotations)
        patch_idx = stride + patch_idx ##simulate the patch movement during conv
        rotations = rotations + stride
        if patch_idx >= dim:
            #print("finished row")
            patch_idx = init_patch_idx
            rotations = stride*dim ##skip stride (vertical) rows

    #for r in m:
    #    print(r)

    return m

"""
kernel_size = 2
stride = 2
input_size = 4
k = np.random.rand(kernel_size,kernel_size)        
m = gen_kernel_matrix(k,kernel_size,stride,input_size)
m = np.array(m)
a = np.random.rand(input_size,input_size)
r = m @ a.flatten()
a_t = torch.from_numpy(a.flatten()).reshape(1,1,input_size,input_size)
k_t = torch.from_numpy(k).reshape(1,1,kernel_size,kernel_size)
c = torch.nn.functional.conv2d(a_t,k_t,stride=stride)
print(c.shape)
print(c.flatten())
print(r)
"""
if __name__ == "__main__":

    ## TEST 1 --> simpleNet simple convolution 
    #with open("./models/simpleNet.json") as f:
    #    model = json.loads(f.read())
    #    k = np.array(model['conv1']['weight']).reshape(5,5,5)
    #    M = []
    #    for kernel in k:
    #        M.append(gen_kernel_matrix(kernel,5,2,29))
    #    data = np.random.rand(29,29)
    #    data_t = torch.from_numpy(data).reshape(1,1,29,29)
    #    k_t = torch.from_numpy(k).reshape(5,1,5,5)
    #    c = torch.nn.functional.conv2d(data_t,k_t,stride=2).reshape(5,13,13)
#
    #    for i,m in enumerate(M):
    #        a = c[i].flatten()
    #        b = m @ data.flatten()
    #        print("Diff between conv and transform_conv:")
    #        print(a-b)

    ## TEST 2 --> alexNet RGB conv kind of
    stride=2
    kernel_size=5
    input_size=29
    in_chans = 3
    out_chans = 64
    output_size = math.floor((input_size - kernel_size)/stride + 1)
    k = np.random.rand(out_chans,in_chans,kernel_size,kernel_size)
    flattened_kernels = []
    for kernel in k:
        ## each kernel has 3 filter, 1 per channel
        #print(kernel.shape)
        flattened_kernel = []
        for channel in kernel:
            m = gen_kernel_matrix(channel,kernel_size=kernel_size,stride=stride,dim=input_size)
            flattened_kernel.append(m)
        flattened_kernels.append(flattened_kernel)
    
    data = np.random.rand(in_chans,input_size,input_size)
    data_t = torch.from_numpy(data).reshape(1,in_chans,input_size,input_size)
    k_t = torch.from_numpy(k).reshape(out_chans,in_chans,kernel_size,kernel_size)
    c = torch.nn.functional.conv2d(data_t,k_t,stride=stride).reshape(out_chans,output_size,output_size)
    loss = []
    for i,fl_k in enumerate(flattened_kernels):
        a = c[i].flatten()
        b = np.zeros((1,output_size*output_size))
        for channel,d in zip(fl_k,data):
            ## compute the linearized conv for each channel and sum'em up
            #print(d.shape)
            #print(f.shape)
            b = b + channel @ d.flatten()
        dist = np.linalg.norm(a.numpy()-b)
        print("conv:",a.numpy())
        print("linear:",b)
        loss.append(dist)
    print("Avg euclidean distance between conv and transform_conv:")
    print(abs(np.average(np.array(loss))))

    ## TEST 3 --> SimpleNet pooling
    ###btw pooling is for free, it is simply element wise multiplication plus sum of all elements, so log(n) rots

    #with open("./models/simpleNet.json") as f:
    #    model = json.loads(f.read())
    #    k = np.array(model['pool1']['weight']).reshape(100,5,13,13)
#
    #    flattened_kernels = []
    #    for kernel in k:
    #        ## each kernel has 5 filter, 1 per channel
    #        #print(kernel.shape)
    #        flattened_kernel = []
    #        for channel in kernel:
    #            m = channel.flatten()
    #            flattened_kernel.append(m)
    #        flattened_kernels.append(flattened_kernel)
    #    
    #    data = np.random.rand(5,13,13)
#
    #    k_t = torch.from_numpy(k).reshape(100,5,13,13)
    #    data_t = torch.from_numpy(data).reshape(1,5,13,13)
    #    c = torch.nn.functional.conv2d(data_t,k_t,stride=1000).reshape(100,1)
    #    loss = []
    #    for i,fl_k in enumerate(flattened_kernels):
    #        a = c[i]
    #        b = 0
    #        for channel,d in zip(fl_k,data):
    #            ## compute the linearized conv for each channel and sum'em up
    #            #print(d.shape)
    #            #print(f.shape)
    #            b = b +  np.sum(channel * d.flatten())
    #        print("conv:",a)
    #        print("linear:",b)
    #        dist = np.linalg.norm(a-b)
    #        loss.append(dist)

    #    print("Avg euclidean distance between conv and transform_conv:")
    #    print(abs(np.average(np.array(loss))))


        