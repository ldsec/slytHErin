import numpy as np
import torch
import math
from collections import deque
import time
import sys
import os
import glob
import json



types = ['weight', 'bias']

def rotR(a,k):
    """
        rot right a k positions
    """
    rot_a = deque(a)
    rot_a.rotate(k)
    return list(rot_a)

"""
    Given a tensor of dimention CxHxDxD returns 2D np array of dim Cx(HxDxD)
    i.e for each input, for each channel of the input, this is row-flattened and then
    all flattened channels are horizontally stacked
"""
def flat_tensor(X):
    #input
    rows, chans, dim1,dim2 = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X_flat = np.zeros((rows, ((dim1*dim2)*chans)))
    for i in range(rows):
            flat = []
            for j in range(chans):
                chan = X[i][j].flatten()
                for c in chan:
                    flat.append(c.item())
            X_flat[i] = np.array(flat)
    return X_flat

"""
    Transform each kernel of a conv layer in a matrix m:
        Conv(input,k) --> m @ input.T.flatten()
        where @ is the usual matrix multiplication
"""
def gen_kernel_matrix(kernel, kernel_size, stride, dim, tranpose=False):
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
        Conv(input,k) --> m @ input.T.flatten()
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

