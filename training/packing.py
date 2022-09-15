import numpy as np
import torch
import math
from collections import deque
import time
import sys
import os
import glob
import json
from conv_transform import *

#####################################
#                                   #
#               PACKING             #
#                                   #
#####################################
#Wrapper for Json dump. Accepts weight dictionary from pack method (pack_conv, pack_linear etc...) and bias dictionary from pack_bias
class Layer:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def Serialize(self):
        return {'weight':self.weight, 'bias':self.bias}
    
# Wrapper for Json dump of net. Accepts layers as list of layers []Layer
class Net:
    def __init__(self, layers, numLayers):
        self.layers = layers
        self.numLayers = numLayers

    def Serialize(self):
        return {'layers': [l.Serialize() for l in self.layers], 'numLayers': self.numLayers}

# Handles packing of a network for HE inference in Go framework
# User has to define custom serializer and packer for its network
# Input:
#   serializer:
#       method customly defined that takes data (e.g model from pytorch, json file...)
#       and outputs a dictionary to be used by packer. The only constraint is that the dictionary is compliant with the packer
#       user has to define its custom serializer and packer
#   packer:
#       takes the customly defined dictionary from serializer
#       the packer should invoke, for every layer (key in the serialized dict)
#       the packing method for the weight (e.g pack_conv, pack_linear or pack_pool)
#       and the packing method pack_bias for the bias of the layer, setting num_channels and replication accordingly
#       output MUST be a class Net in serialized form,
#       i.e as a rule of thumb you should define a Net() in your packer method and return Net.Serialize()
class Packer:
    def __init__(self, serializer, packer):
        self.serializer = serializer
        self.packer = packer
    
    def Pack(self, net):
        serialized = self.serializer(net)
        return self.packer(serialized)

"""
    The idea:
        gen_kernel_matrix(k) returns for each channel of a kernel in a conv layer, a matrix m s.t
        m @ x.T = conv(k,x) (where @ is the matrix multiplication)

        if we have n kernels with f channels (meaning that input image has f dimensions),
        we can generate a matrix M

        M = | m(k1_ch1) |...| m(k1_chf)|
            | m(k2_ch1) |...| m(k2_chf)|
            | .........................|
            | m(kn_ch1) |...| m(kn_chf)|

        s.t 

        M @ X.T = conv(k,X)
        where each row of x is a flattened data sample |x_1|...|x_f| (x_i is the channel i of one image)

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
"""

"""
    Pack kernel matrixes, each kernel filter in Toepltiz representation
    Output is:
    |k1_ch1|...|k1_chn|
    |.................|
    |.................| --> tranposed
    |.................|
    |km_ch1|...|km_chn|
    for a conv layer with n input channels and m output channels
    Input is a conv layer as a np tensor of dim KxFxRxC with
    K kernels, each with
    F filters, which are
    R x C matrices
"""
def pack_conv(conv, kernel_rows, kernel_cols, stride, input_rows, input_cols):
    kernel_matrices = []
    for kernel in conv:
        channel_matrices = []
        for channel in kernel:
            if kernel_rows == kernel_cols and input_rows == input_cols:
                m = gen_kernel_matrix(channel,kernel_rows, stride, input_rows)
            else:
                m = gen_kernel_matrix_rect(channel,kernel_rows, kernel_cols, stride, input_rows, input_cols)
            channel_matrices.append(np.array(m))
        channelM = np.hstack(channel_matrices)
        kernel_matrices.append(channelM)
    convM = np.vstack(kernel_matrices)
    rows, cols = convM.T.shape
    return {'w': convM.T.flatten().tolist(), 'rows': rows, 'cols': cols}, convM.T

"""
    For cryptonet
    Similar to pack_conv, we take advantage that kernel filter has same size of input:
"""
def pack_pool(pool):
    kernel_matrices = []
    for kernel in pool:
        channels = []
        for channel in kernel:
            channels.append(channel.flatten().T)
        channel_matrix = np.vstack(channels).flatten().T
        kernel_matrices.append(channel_matrix)
    poolM = np.vstack(kernel_matrices)
    rows, cols = poolM.T.shape
    return {'w': poolM.T.flatten().tolist(), 'rows': rows, 'cols': cols}, poolM.T

"""
    Packs a linear layer. the matrix should be s.t the evaluation of the layer
    is
    W @ X + b, so you have to transpose first is needed (for example from pythorch nn.Linear)
"""
def pack_linear(dense):
    rows, cols = dense.shape
    return {'w': dense.flatten().tolist(), 'rows': rows, 'cols': cols}

"""
        Pack bias as an array to be summed to the convolution of one data sample
        Since in our packing the output channels are all on the same row, bias values
        are replicated size of conv output size for each of the channels,
        i.e replication is: 
        cols of packed conv layer // num of channels
        E.g:
            given
                bias = [b1,...,bn]

                x = |x1 * k1|...|x1 * kn|
            packed is:
                b = |b1...b1|...|bn...bn| --> this to be replicated for every row of a matrix of rows batch size
        If layer is dense, set replication to 1 and channels as the number of cols of the weight
"""
def pack_bias(b, channels, replication):
    #dense
    if replication == 1:
        return {'b': b.tolist(), 'len': channels}
    #conv
    bias = [0 for i in range(channels*replication)]
    for i in range(channels*replication):
        idx = i // replication
        bias[i] = b.item(idx)
    return {'b': bias, 'len': channels*replication}