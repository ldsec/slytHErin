import numpy as np
import torch
import math
from collections import deque
import time
import sys
import os
import glob
import json

"""
    This script, and its _test variant, contain test evaluated with the original model in pytorch
    of different linear approximations of the convolutional layers
    to be evalauted in the inference directory, in Go, under
    homomorphic encryption

    Python allows developing quick PoC for various methods, and also allows
    quick packaging of matrix with numpy agility

    The idea:
        gen_kernel_matrix(k) returns for each channel of a kernel, a matrix m s.t
        m @ x.T = conv(k,x)

        if we have n kernels with f channels (meaning that input image has f dimensions),
        we can generate a matrix M

        M = | m(k1_ch1) |...| m(k1_chf)|
            | m(k2_ch1) |...| m(k2_chf)|
            | .........................|
            | m(kn_ch1) |...| m(kn_chf)|

        s.t 

        X @ M.T = conv(k,X)
        where each row of x is a flattened data sample |x_1|...|x_f|

        The output is going to be a matrix b x (output_dim**2)*n
        where n is the number of output channels (i.e kernels)
        and output_dim is the dimention of a single convolution between x_i and a channel j of a kernel i,
        so 
        X @ M.T =
         |x1 * k1|...|x1 * kn|
         |x2 * k1|...|x2 * kn|
         |...................|
         |xb * k1|...|xb * kn| 

        Following this, is easy to pack the subsequent layers as the output format is consistent with the input

        Also, there are parallelized version of the methods to provide smaller matrixes to carry out
        the pipeline with smaller ciphers --> not used in Go implementation
"""

types = ['weight', 'bias']

def rotR(a,k):
    """
        rot right a k positions
    """
    rot_a = deque(a)
    rot_a.rotate(k)
    return list(rot_a)

def flat_tensor(X):
    """
        Given a tensor of dimention CxHxDxD returns 2D np array of dim Cx(HxDxD)
    """
    rows, chans, dim = X.shape[0], X.shape[1], X.shape[2]
    X_flat = np.zeros((rows, ((dim)**2)*chans))
    for i in range(rows):
            flat = []
            for j in range(chans):
                chan = X[i][j].flatten()
                for c in chan:
                    flat.append(c)
            X_flat[i] = np.array(flat)
    return X_flat

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

def gen_kernel_matrix_rect(kernel, kernel_rows, kernel_cols, stride, input_rows, input_cols, tranpose=False):
    """
    Transform each kernel of a conv layer in a matrix m:
        Conv(input,k) --> m @ input.flatten()
        where @ is the usual matrix multiplication

    Handles non square matrixes
    """
    out_rows = math.floor((input_rows - kernel_rows)/stride + 1)
    out_cols = math.floor((input_cols - kernel_cols)/stride + 1)
    ## create a matrix of dimention nxm, where:
    ##  n is the lenght of flattened output
    ##  m is the dim of flattened input
    m = [[0 for i in range(input_rows*input_cols)] for j in range(out_rows*out_cols)]
    kernel_flattened = m[0] ## 0 vector of dim^2
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            ## fill the vector by assigning each kernel weight to
            ## its position in the input matrix in the first patch
            ## we will then rotate this
            kernel_flattened[i*input_cols + j] = kernel[i][j]

    patch_idx = kernel_cols-1
    init_patch_idx = patch_idx ## horizontal index
    rotations = 0
    row = 0 #which row in input
    for i in range(len(m)):
        v = rotR(kernel_flattened, rotations)
        m[i] = v
        patch_idx = stride + patch_idx ##simulate the patch movement during conv
        rotations = rotations + stride
        if patch_idx >= input_cols:
            patch_idx = init_patch_idx
            rotations = 0
            rotations = row*input_cols + stride*input_cols ## the row we are now + skip stride (vertical) rows
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


#####################################
#                                   #
#               PACKING             #
#                                   #
#####################################

"""
    Please ignore the parallel version of methods as they are not used!
"""

def pack_conv(conv, kernel_size, stride, input_dim):
    """
        Pack kernel matrixes, each kernel filter in Toepltiz representation
        Output is:
        |k1_ch1|...|k1_chn|
        |.................|
        |.................| --> tranposed
        |.................|
        |km_ch1|...|km_chn|
        for a conv layer with n input channels and m output channels
    """
    kernel_matrices = []
    for kernel in conv:
        channel_matrices = [] 
        for channel in kernel:
            m = gen_kernel_matrix(channel,kernel_size=kernel_size,stride=stride,dim=input_dim)
            channel_matrices.append(np.array(m))
        channelM = np.hstack(channel_matrices)
        kernel_matrices.append(channelM)
    convM = np.vstack(kernel_matrices)
    rows, cols = convM.T.shape
    return {'w': [x.item() for x in convM.T.flatten()], 'rows': rows, 'cols': cols}, convM.T

def pack_conv_rect(conv, kernel_rows, kernel_cols, stride, input_rows, input_cols):
    """
        Same as conv, but with non square matrices.

        Pack kernel matrixes, each kernel filter in Toepltiz representation
        Output is:
        |k1_ch1|...|k1_chn|
        |.................|
        |.................| --> tranposed
        |.................|
        |km_ch1|...|km_chn|
        for a conv layer with n input channels and m output channels
    """
    kernel_matrices = []
    for kernel in conv:
        channel_matrices = [] 
        for channel in kernel:
            m = gen_kernel_matrix_rect(channel,kernel_rows, kernel_cols, stride, input_rows, input_cols)
            channel_matrices.append(np.array(m))
        channelM = np.hstack(channel_matrices)
        kernel_matrices.append(channelM)
    convM = np.vstack(kernel_matrices)
    rows, cols = convM.T.shape
    return {'w': [x.item() for x in convM.T.flatten()], 'rows': rows, 'cols': cols}, convM.T

def pack_conv_parallel(conv, kernel_size, stride, input_dim):
    """
        returns f lists of n matrices, according to input channels and output channels of this layer.
        Example if this is pool1 of SimpleNet, it has 5 input and 100 output channels.
        Each list will be:
        [m(ch_1),...,m(ch_5)] for i = 1,...,100 (# kernels)
        where m(ch_1) @ X.T is X * ch_1 where * is convolution
    """
    kernels = []
    for kernel in conv: #output_chan
        filters = []
        for filter in kernel: #input chan
            m = gen_kernel_matrix(filter,kernel_size=kernel_size,stride=stride,dim=input_dim)
            filters.append(np.array(m))
        kernels.append(filters)
    return kernels

def pack_pool(pool):
    """
        For simplenet
        Similar to pack_conv, we take advantage that kernel filter has same size of input:
    """
    kernel_matrices = []
    for kernel in pool:
        channels = []
        for channel in kernel:
            channels.append(channel.flatten().T)
        channel_matrix = np.vstack(channels).flatten().T
        kernel_matrices.append(channel_matrix)
    poolM = np.vstack(kernel_matrices)
    rows, cols = poolM.T.shape
    return {'w': [x.item() for x in poolM.T.flatten()], 'rows': rows, 'cols': cols}, poolM.T

def pack_linear(dense):
    """
        Packs a linear layer
    """
    rows, cols = dense.shape
    return {'w': [x.item() for x in dense.flatten()], 'rows': rows, 'cols': cols}

def pack_linear_parallel(dense, num_channels):
    """
        We transform layer to convolutional for smoothing the transition between conv and linear
    """
    num_kernels = dense.shape[1]
    size = dense.shape[0] // num_channels #size of input image squared
    conv = []
    for i in range(num_kernels):
        channels = []
        for j in range(num_channels):
            channels.append(np.array(dense[j*size:(j+1)*size][i]))
        conv.append(channels)
    return conv

def pack_bias(b, channels, replication):
    """
        Pack bias as an array to be summed to the convolution of one data sample
        Since in our packing the output channels are all on the same row, bias values
        are replicated size of conv output size for each of the channels
        E.g:
            given
                bias = [b1,...,bn]
                
                x = |x1 * k1|...|x1 * kn|
            packed is:
                b = |b1...b1|...|bn...bn| --> this to be replicated for every row of a matrix of rows batch size
    """
    bias = [0 for i in range(channels*replication)]
    for i in range(channels*replication):
        idx = i // replication
        bias[i] = b[idx].item()
    return {'b': bias, 'len': channels*replication}

def pack_bias_parallel(b, size):
    """
        Given b = [b1,...,bn]
        returns:
        <--size-->
        [[b1,...,b1],
        ...
        [bn,...,bn]]
        where each bi vector of <size>-len is tranposed to be compatible with the gen_kernel_matrix format
    """
    return [np.array([b[i] for _ in range(size)]).T for i in range(len(b))]
    
def compress_layers(A, biasA, B, biasB):
    """
    Compress two linear layers back to back:
    x = x.A + biasA
    x = x.B + biasB

    x = (x.A + biasA).B + biasB = xA.B + biasA.B+biasB
    returns:
    AB, biasAB + biasB

    inputs are packed bias and layer matrix tranposed as per pack methods
    """
    C = A @ B
    biasC = np.array(biasA['b']) @ B
    if biasB != None:
        biasC = biasC + np.array(biasB['b'])
    l = [x.item() for x in biasC]
    return C, {'b':l, 'len':len(l)}

def compress_layers_parallel(A,biasA,B,biasB):
    """
        compress layers in parallel form
        here we have B(A @ X.T + biasA)+biasB
        Assume that A has d input channels and n output channels
        Assume that A is followed by B
        B takes n input channels and m output channels
        Result will have d input and m output channels and m bias values
    """
    output_channels = len(B)
    mid_channels = len(A)
    if mid_channels != len(B[0]):
        sys.exit("Fatal: parallel layer dim mismatch")
    input_channels = len(A[0])

    output = [[] for _ in range(output_channels)]
    for i in range(output_channels):
        for j in range(input_channels):
            tmp = B[0][0] @ A[0][0]
            for k in range(1,mid_channels):
                tmp = tmp + B[i][k] @ A[k][j]
            output[i].append(tmp)
    if biasA != None:
        bias = []
        for i in range(output_channels):
            tmp = B[0][0] @ biasA[0]
            for j in range(1,mid_channels):
                tmp = tmp + B[i][j] @ biasA[j]
            bias.append(tmp + biasB[i])
    else:
        bias = biasB

    return output, bias

def gen_padding_matrix(input_dim, chans, pad):
    '''
    Generate a sparse matrix that adds padding to the result of a convolutional layer
    (i.e the one where each row is a flattened sample with adjacent channels)
    '''
    cols = chans*(input_dim**2)
    rows = chans*(2*pad*(2*pad+input_dim) + input_dim*(input_dim+2*pad))

    eyepool = [] #list of matrices
    #block = np.zeros((input_dim,input_dim))
    #eye = np.eye(input_dim, dtype=float)
    # we are sliding blocks of width input_dim, so divide c per input dim
    for i in range(chans*input_dim):
        tmp = [[] for _ in range(input_dim)]
        for r in range(input_dim):
            tmp[r] = [0 for _ in range(chans*input_dim*input_dim)]
        for r in range(input_dim):
            tmp[r][r+i*input_dim] = 1
        eyepool.append(tmp)
        #for j in range(1, len(tmp)):
        #    eyepool[-1] = np.hstack((eyepool[-1],tmp[j]))

    #rowpad = np.zeros((pad*(input_dim+pad*2), cols))
    #colpad = np.zeros((pad,cols))
    rowpad = pad*(input_dim+pad*2)
    colpad = pad

    P = [[] for _ in range(rows)]
    for r in range(rows):
        P[r] = [0 for _ in range(cols)]
    
    pool_counter = 0
    idx = 0
    for _ in range(chans):
        idx = idx + rowpad
        for row in range(input_dim):
            idx = idx + colpad
            for row_of_eye in range(len(eyepool[0])):
                P[idx] = eyepool[pool_counter][row_of_eye]
                idx = idx + 1
            pool_counter = pool_counter + 1
            idx = idx + colpad
        idx = idx + rowpad
    #for ch in range(chans):
    #    if skip != True:
    #        P = np.vstack((P, rowpad))
    #    else:
    #        skip = False
    #    for row in range(input_dim):
    #        P = np.vstack((P, colpad))
    #        P = np.vstack((P, eyepool[pool_counter]))
    #        pool_counter = pool_counter + 1
    #        P = np.vstack((P, colpad))
    #    P = np.vstack((P, rowpad))

    return np.array(P).T
    

def pad_parallel(P, layer):
    for i in range(len(layer)):
        for j in range(len(layer[0])):
            layer[i][j] = P @ layer[i][j]
    return layer


    
