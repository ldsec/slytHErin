import numpy as np
import torch
import math
from collections import deque
from activation import relu_approx, sigmoid_approx
from dataHandler import DataHandler, DataHandlerAlex
from cryptonet import SimpleNet
from alexnet import AlexNet
import time
import sys
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

## JP function
#def gen_kernel_matrix_JP(kernel, stride_h, stride_v, dim):
#    m = [[0 for i in range(dim*dim)] for j in range(dim*dim)]
#
#    kernel_flattened = [0 for i in range(len(kernel) * dim - (dim-len(kernel)))]
#
#    for i in range(len(kernel)):
#        for j in range(len(kernel)):
#            kernel_flattened[i*dim + j] = kernel[i][j]
#    a = (dim - len(kernel) + stride_v)//stride_v
#    b = (dim - len(kernel) + stride_h)//stride_h
#
#    for i in range(a):
#        for j in range(b):
#            idx_j = j * stride_h + i * stride_v * dim
#            for k in range(len(kernel_flattened)):
#                m[i*b+j][idx_j+k] = kernel_flattened[k]
#    #for i in m:
#    #    print(i)
#    return m
#

def rotR(a,k):
    """
        rot right a k positions
    """
    rot_a = deque(a)
    rot_a.rotate(k)
    return list(rot_a)

def flat_tensor(X):
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
'''
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
'''
def gen_kernel_matrix(kernel, kernel_rows, kernel_cols, stride, input_rows, input_cols, tranpose=False):
    """
    Transform each kernel of a conv layer in a matrix m:
        Conv(input,k) --> m @ input.flatten()
        where @ is the usual matrix multiplication

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

def extract_param(param_name, param):
    '''
        Params are extracted in row-major order:
        suppose you have a CONV layer with (k,C,W,H), 
        i.e k kernels wich are tensors of dim C x W x H
        each kernel is flattened such that every W x H matrix
        is flattened by row, for C matrixes. This for every k kernel 
    '''
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
    if 'bias' in param_name:
        weights = []
        data = param.data.cpu().numpy().flatten()
        for k in data:
            weights.append(k.item())
    return weights

def serialize_model(model, format):
    '''
    Serialize the model. Returns a dictionary which maps layer name to a flattened
    representation of the underlying weight in a json-serializable format
    '''
    serialized = {}
    for name, p in model.named_parameters():
        serialized[name] = extract_param(name,p)
    serialized = format(serialized)
    return serialized

'''
Format methods just rearrange the serialized representation of models
'''
def format_AlexNet(serialized):
    formatted = {}
    ## get all layers name
    layers = []
    for k in serialized.keys():
        l = k.split(".")
        if l[0] == 'classifier': ##this is the nn.Sequential layer
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

#####################################
#                                   #
#               PACKING             #
#                                   #
#####################################

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


def pack_conv_parallel(conv, kernel_size, stride, input_dim):
    """
        returns f lists of n matrices, according to input channels and output channels of this layer.
        Example if this is pool1 of SimpleNet, it has 5 input and 100 output channels.
        Each list will be:
        [m(ch_1),...,m(ch_5)] for i = 1,...,100 (# kernels)
        where m(ch_1) @ X.T is X * ch_1
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
    

'''
Pack methods: return model in json form ready to be shipped to Go implementation
in an HE friendly format
'''
def pack_simpleNet(model):
    serialized = serialize_model(model, format_SimpleNet)
    
    conv1 = np.array(serialized['conv1']['weight']).reshape(5,1,5,5)
    pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,13,13)
    pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    b1 = np.array(serialized['conv1']['bias'])
    b2 = np.array(serialized['pool1']['bias'])
    b3 = np.array(serialized['pool2']['bias'])

    conv1M,_ = pack_conv(conv1,5,2,29)
    bias1 = pack_bias(b1, 5, 13*13)

    pool1M,_ = pack_pool(pool1)
    bias2 = pack_bias(b2, 100, 1)

    pool2M,_ = pack_pool(pool2)
    bias3 = pack_bias(b3, 10, 1)

    packed = {}
    packed['conv1'] = {'weight':conv1M, 'bias': bias1}
    packed['pool1'] = {'weight':pool1M, 'bias': bias2}
    packed['pool2'] = {'weight':pool2M, 'bias': bias3}
    return packed

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

def test_pad():
    input_dim = 20
    chans = 64
    pad = 2
    l = []
    l_pad = []
    for i in range(chans):
        tmp = np.random.rand(input_dim, input_dim)
        tmp_pad = np.pad(tmp, pad, 'constant', constant_values=0)
        l.append(list(tmp.flatten())) #tmp.flatten()
        l_pad.append(list(tmp_pad.flatten()))
    
    A = []
    A_pad = []
    for i in range(len(l)):
        for j in range(len(l[0])):
            A.append(l[i][j])
    for i in range(len(l_pad)):
        for j in range(len(l_pad[0])):
            A_pad.append(l_pad[i][j])
    A = np.array(A)
    A_pad = np.array(A_pad)
    print("Creating pad")
    start = time.time()
    P = gen_padding_matrix(input_dim,chans,pad)
    print("Done")
    print(time.time()-start)
    print("Computing the mul")
    Ap = P @ A.T
    
    print("Dist")
    print(np.linalg.norm(Ap-A_pad))

def assemble_layer_from_matrix(M):
    rows, cols = M.shape
    return {'w': [x.item() for x in M.flatten()], 'rows': rows, 'cols': cols}

def conv_parallel(conv,bias, X):
    Y = []
    for sample in X:
        sample_channels = []
        if bias != None:
            for b,kernel in zip(bias,conv):
                tmp = sample[0] @ kernel[0].T 
                for channel in range(1,len(kernel)):
                    tmp = tmp + sample[channel] @ kernel[channel].T
                tmp = tmp + b
                sample_channels.append(tmp)
        else:
            for kernel in conv:
                tmp = sample[0] @ kernel[0].T 
                for channel in range(1,len(kernel)):
                    tmp = tmp + sample[i] @ kernel[channel].T
                sample_channels.append(tmp)

        Y.append(sample_channels)

    return Y

def activate_parallel(X, activation):
    for sample in X:
        for channel in sample:
            channel = activation(torch.from_numpy(channel)).numpy()
    return X

def pack_alexNet(model):
    serialized = serialize_model(model, format_AlexNet)

    # reshape(chan_out, chan_in, k_size,k_size)
    conv1 = np.array(serialized['conv1']['weight']).reshape(64,3,11,11)
    conv2 = np.array(serialized['conv2']['weight']).reshape(192,64,5,5)
    conv3 = np.array(serialized['conv3']['weight']).reshape(384,192,3,3)
    conv4 = np.array(serialized['conv4']['weight']).reshape(256,284,3,3)
    conv5 = np.array(serialized['conv5']['weight']).reshape(256,256,3,3)
    #pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,13,13)
    #pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    pool1 = np.ones((256,256,3,3))*(1.0/(3**2))
    pool2 = np.ones((256,256,3,3))*(1.0/(3**2))
    dense1 = np.array(serialized['classifier.1']['weight']).reshape(1,1,9216,4096)
    dense2 = np.array(serialized['classifier.4']['weight']).reshape(1,1,4096,4096)
    dense3 = np.array(serialized['classifier.6']['weight']).reshape(1,1,4096,10)


    bias_conv1 = np.array(serialized['conv1']['bias'])
    bias_conv2 = np.array(serialized['conv2']['bias'])
    bias_conv3 = np.array(serialized['conv3']['bias'])
    bias_conv4 = np.array(serialized['conv4']['bias'])
    bias_conv5 = np.array(serialized['conv5']['bias'])
    bias_dense1 = np.array(serialized['classifier.1']['bias'])
    bias_dense2 = np.array(serialized['classifier.4']['bias'])
    bias_dense3 = np.array(serialized['classifier.6']['bias'])

    #linearize layers
    #conv1MD, conv1MT = pack_conv(conv1,11,4,229)
    conv1M = pack_conv_parallel(conv1, 11, 4, 227+2*2)
    bias_conv1M = pack_bias_parallel(bias_conv1, 56*56)

    pool1AM = pack_conv_parallel(pool1,3,2,56)

    conv2M= pack_conv_parallel(conv2,5,1,27+2*2)
    bias_conv2M = pack_bias_parallel(bias_conv2,  27*27)

    pool1BM = pack_conv_parallel(pool1,3,2,27)

    conv3M = pack_conv_parallel(conv3,3,1,13+1*2)
    bias_conv3M = pack_bias_parallel(bias_conv3, 13*13)
    
    conv4M = pack_conv_parallel(conv4,3,1,13+1*2)
    bias_conv4M = pack_bias_parallel(bias_conv4, 13*13)

    conv5M = pack_conv_parallel(conv5,3,1,13+1*2)
    bias_conv5M = pack_bias_parallel(bias_conv5, 13*13)

    pool1CM = pack_conv_parallel(pool1,3,2,13)

    pool2M = pack_conv_parallel(pool2,3,1,6+1*2)

    #compress layers
    P = gen_padding_matrix(25,1,2)
    pool1AM = pad_parallel(P, pool1AM)
    pool1_conv2M, bias_pool1_conv2 = compress_layers(pool1AM, None, conv2M, bias_conv2M)

    P = gen_padding_matrix(13,1,1)
    pool1BM = pad_parallel(P, pool1BM)
    pool1_conv3M, bias_pool1_conv3 = compress_layers(pool1BM, None, conv3M, bias_conv3M)
    
    P = gen_padding_matrix(13,1,1)
    conv4M = pad_parallel(P, conv4M)
    conv5M = pad_parallel(P, conv5M)

    P = gen_padding_matrix(6,1,1)
    pool1CM = pad_parallel(P, pool1CM)
    pool1_pool2, bias_pool1_pool2 = compress_layers(pool1CM, None , pool2M, None)

    packed = {}
    #packed['conv1'] = {'weight':conv1MD, 'bias': bias_conv1M}
    #packed['conv2'] = {'weight':pool1_conv2MD, 'bias': bias_pool1_conv2}
    #packed['conv3'] = {'weight':pool1_conv3MD, 'bias': bias_pool1_conv3}
    #packed['conv4'] = {'weight':conv4MD, 'bias': bias_conv4M}
    #packed['conv5'] = {'weight':conv5MD, 'bias': bias_conv5M}
    #packed['pool'] = {'weight':pool1_pool2MD, 'bias': bias_pool1_pool2}
    #packed['dense1'] = {'weight':pack_linear(dense1), 'bias':{'b': [x.item() for x in bias_dense1], 'len':len(bias_dense1)}}
    #packed['dense2'] = {'weight':pack_linear(dense2), 'bias':{'b': [x.item() for x in bias_dense2], 'len':len(bias_dense2)}}
    #packed['dense3'] = {'weight':pack_linear(dense3), 'bias':{'b': [x.item() for x in bias_dense3], 'len':len(bias_dense3)}}

    return packed

if __name__ == "__main__":
    ### TEST 0 --> conv with rect kernel
    data = np.random.rand(2,28,28)
    kernel = np.random.rand(2,11,10)
    M = []
    for k in kernel:
        M.append(gen_kernel_matrix(k,11,10,2,28,28,True))

    data_t = torch.from_numpy(data).reshape(2,1,28,28)
    k_t = torch.from_numpy(kernel).reshape(2,1,11,10)
    c = torch.nn.functional.conv2d(data_t,k_t,stride=2, groups=1)
    loss = []
    data = data.reshape(2,28*28)
    for j in range(2):
        for i,m in enumerate(M):
            a = c[j][i].flatten()
            a = a.flatten()
            b = data @ m
            dist = np.linalg.norm(a.numpy()-b[j].flatten())
            loss.append(dist)
    print("Avg euclidean distance between conv and transform_conv:")
    print(abs(np.average(np.array(loss))))

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
    
    ##TEST 4 -> simplenet pipeline (working fine)
    #device = torch.device('cpu')
    ##torch_model = SimpleNet(64, 'relu_approx', 'none', 'xavier', False)
    ##torch_model.load_state_dict(torch.load("./models/SimpleNet_xavier_relu_approx.pt",map_location=device))
    #torch_model = torch.load("./models/SimpleNet_xavier_relu_approx.pt",map_location=device)
    #torch_model = torch_model.double()
    #torch_model.batch_size = 64
    #torch_model.eval()
    #model = serialize_model(torch_model, format_SimpleNet)
    #conv1 = np.array(model['conv1']['weight']).reshape(5,1,5,5)
    #pool1 = np.array(model['pool1']['weight']).reshape(100,5,13,13)
    #pool2 = np.array(model['pool2']['weight']).reshape(10,1,100,1)
    #b1 = np.array(model['conv1']['bias'])
    #b2 = np.array(model['pool1']['bias'])
    #b3 = np.array(model['pool2']['bias'])
    #dh = DataHandler("MNIST", 64, False)
    #correct_conv = 0
    #correct_linear = 0
    #correct_pytorch = 0
    #tot = 0
    #
    #for X,Y in dh.test_dl:
    #    X = X.double()
    #    X_torch = X
    #    X = torch.nn.functional.pad(X,(1,0,1,0))
    #    X_t = X
    #    X = X_t.numpy()
    #    X_flat = np.zeros((64,841))
    #    
    #    ## conv 1
    #    for i in range(64):
    #        X_flat[i] = X[i][0].flatten()
    #    c1 = torch.nn.functional.conv2d(X_t,torch.from_numpy(conv1),bias=torch.from_numpy(b1),stride=2)
    #    _, conv1M = pack_conv(conv1, 5,2,29)
    #    d1 = X_flat @ conv1M
    #    ##parallelized
    #    conv1_matrixes = pack_conv_kernel_parallel(conv1, 5, 2, 29)
    #    d1_parallel = []
    #    for k in conv1_matrixes:
    #        d1_parallel.append(X_flat @ k)
    #    bias1 = pack_bias(b1, 5, 13*13)
    #    for i in range(d1.shape[0]):
    #        d1[i] = d1[i] + bias1['b']
    #    for i,d1_p in enumerate(d1_parallel):
    #        bias = np.ones(len(d1_p[0]))*b1[i]
    #        bias[-1] = 0
    #        d1_p = d1_p + bias
    #    
    #    c1_flat = np.zeros(d1.shape)
    #    for i,c in enumerate(c1):
    #        r = []
    #        for ch in c:
    #            r.append(ch.flatten())
    #        c1_flat[i] = np.hstack(r)
    #    dist = np.linalg.norm(c1_flat-d1)
    #    print("conv1", dist)
#
    #    ##pool1
#
    #    c2 = torch.nn.functional.conv2d(c1, torch.from_numpy(pool1),bias=torch.from_numpy(b2),stride=1000)
    #    c2 = relu_approx(c2)
    #    c2_f = c2.reshape(c2.shape[0],-1)
    #    _, pool1M = pack_pool(pool1)
    #    d2 = d1 @ pool1M
#
    #    ##parallelized
    #    pool1_matrixes = pack_pool_channels_parallel(pool1)
    #    l = []
    #    for i in range(len(d1_parallel)):
    #        l.append(d1_parallel[i] @ pool1_matrixes[i])
    #    d2_p = np.zeros(l[0].shape)
    #    for i in range(len(l)):
    #        d2_p = d2_p + l[i]
#
    #    bias = pack_bias(b2,100,1)
    #    for i in range(d2.shape[0]):
    #        d2[i] = d2[i] + bias['b']
    #        d2_p[i] = d2_p[i] + bias['b']
    #    d2 = relu_approx(torch.from_numpy(d2)).numpy()
    #    d2_p = relu_approx(torch.from_numpy(d2_p)).numpy()
    #    dist = np.linalg.norm(c2_f.numpy()-d2_p)
    #    print("pool1", dist)
#
    #    ##pool2
#
    #    c3 = torch.nn.functional.conv2d(c2.reshape(64,1,100,1), torch.from_numpy(pool2),bias=torch.from_numpy(b3), stride=1000)
    #    c3 = relu_approx(c3)
    #    c3 = c3.reshape(c3.shape[0],-1)
    #    _, pool2M = pack_pool(pool2)
    #    d3 = d2 @ pool2M
#
    #    ##parallelized
    #    pool2_matrixes = pack_pool_channels_parallel(pool2)
    #    d3_p = d2_p @ pool2_matrixes[0]
#
    #    bias = pack_bias(b3,10,1)
    #    for i in range(d2.shape[0]):
    #        d3[i] = d3[i] + bias['b']
    #        d3_p[i] = d3_p[i] + bias['b']
    #    d3 = relu_approx(torch.from_numpy(d3)).numpy()
    #    d3_p = relu_approx(torch.from_numpy(d3_p)).numpy()
    #    dist = np.linalg.norm(c3.numpy()-d3_p)
    #    print("pool2", dist)
    #    #print(d3.shape)
    #    _,pred = torch_model(X_torch.double()).max(1)
    #    _,pred_c = c3.max(1)
    #    pred_l = np.argmax(d3_p,axis=1)
    #    correct_pytorch = correct_pytorch + (pred == Y).sum().item()
    #    correct_conv = correct_conv + (pred_c == Y).sum().item()
    #    correct_linear = correct_linear + np.sum(pred_l == Y.numpy())
    #    tot = tot + Y.shape[0]
    #print(f"Accuracy torch {correct_pytorch/tot}")
    #print(f"Accuracy conv {correct_conv/tot}")
    #print(f"Accuracy linear {correct_linear/tot}")
    """
    ## TEST 5 -> AlexNet simplified (what about the padding in between?)
    device = torch.device('cpu')
    #torch_model = SimpleNet(64, 'relu_approx', 'none', 'xavier', False)
    #torch_model.load_state_dict(torch.load("./models/SimpleNet_xavier_relu_approx.pt",map_location=device))
    torch_model = torch.load("./models/AlexNet_simplified.pt",map_location=device)
    torch_model = torch_model.double()
    torch_model.simplified = True
    torch_model.eval()
    
    serialized = serialize_model(torch_model, format_AlexNet)

    # reshape(chan_out, chan_in, k_size,k_size)
    conv1 = np.array(serialized['conv1']['weight']).reshape(64,3,11,11)
    conv2 = np.array(serialized['conv2']['weight']).reshape(192,64,5,5)
    conv3 = np.array(serialized['conv3']['weight']).reshape(384,192,3,3)
    conv4 = np.array(serialized['conv4']['weight']).reshape(256,384,3,3)
    conv5 = np.array(serialized['conv5']['weight']).reshape(256,256,3,3)
    #pool1 = np.array(serialized['pool1']['weight']).reshape(100,5,13,13)
    #pool2 = np.array(serialized['pool2']['weight']).reshape(10,1,100,1)
    pool1 = np.ones((256,256,3,3))*(1.0/(3**2))
    pool2 = np.ones((256,256,3,3))*(1.0/(3**2))
    dense1 = np.array(serialized['classifier.1']['weight']).reshape(1,1,9216,4096)
    dense2 = np.array(serialized['classifier.4']['weight']).reshape(1,1,4096,4096)
    dense3 = np.array(serialized['classifier.6']['weight']).reshape(1,1,4096,10)


    bias_conv1 = np.array(serialized['conv1']['bias'])
    bias_conv2 = np.array(serialized['conv2']['bias'])
    bias_conv3 = np.array(serialized['conv3']['bias'])
    bias_conv4 = np.array(serialized['conv4']['bias'])
    bias_conv5 = np.array(serialized['conv5']['bias'])
    bias_dense1 = np.array(serialized['classifier.1']['bias'])
    bias_dense2 = np.array(serialized['classifier.4']['bias'])
    bias_dense3 = np.array(serialized['classifier.6']['bias'])

    #linearize layers
    #conv1MD, conv1MT = pack_conv(conv1,11,4,229)
    conv1M = pack_conv_parallel(conv1, 11, 4, 227+2*2)
    bias_conv1M = pack_bias_parallel(bias_conv1, 56*56)

    pool1AM = pack_conv_parallel(pool1,3,2,56)

    conv2M= pack_conv_parallel(conv2,5,1,27+2*2)
    bias_conv2M = pack_bias_parallel(bias_conv2,  27*27)

    pool1BM = pack_conv_parallel(pool1,3,2,27)

    conv3M = pack_conv_parallel(conv3,3,1,13+1*2)
    bias_conv3M = pack_bias_parallel(bias_conv3, 13*13)
    
    conv4M = pack_conv_parallel(conv4,3,1,13+1*2)
    bias_conv4M = pack_bias_parallel(bias_conv4, 13*13)

    conv5M = pack_conv_parallel(conv5,3,1,13+1*2)
    bias_conv5M = pack_bias_parallel(bias_conv5, 13*13)

    pool1CM = pack_conv_parallel(pool1,3,2,13)

    pool2M = pack_conv_parallel(pool2,3,1,6+1*2)

    dense1M = pack_linear_parallel(dense1, 256)
    dense2M = pack_linear_parallel(dense2, 256)
    dense3M = pack_linear_parallel(dense3, 256)

    #compress layers
    P = gen_padding_matrix(25,1,2)
    pool1AM = pad_parallel(P, pool1AM)
    pool1_conv2M, bias_pool1_conv2 = compress_layers(pool1AM, None, conv2M, bias_conv2M)

    P = gen_padding_matrix(13,1,1)
    pool1BM = pad_parallel(P, pool1BM)
    pool1_conv3M, bias_pool1_conv3 = compress_layers(pool1BM, None, conv3M, bias_conv3M)
    
    P = gen_padding_matrix(13,1,1)
    conv4M = pad_parallel(P, conv4M)
    conv5M = pad_parallel(P, conv5M)

    P = gen_padding_matrix(6,1,1)
    pool1CM = pad_parallel(P, pool1CM)
    pool1_pool2, bias_pool1_pool2 = compress_layers(pool1CM, None , pool2M, None)


    dh = DataHandlerAlex("MNIST", 64, False)
    correct_linear = 0
    correct_pytorch = 0
    tot = 0
    print("Start inference")
    for X,Y in dh.test_dl:
        X = X.double()
        X_torch = X
        X = torch.nn.functional.pad(X,(2,2,2,2))
        X_t = X
        X = X_t.numpy().reshape(64,3,227+2*2,227+2*2)
        X_flat = [[X[i][j].flatten() for j in range(X.shape[1])] for i in range(X.shape[0])]
        
        ##conv1
        d1 = conv_parallel(conv1M, bias_conv1M, X_flat)
        d1 = activate_parallel(d1, relu_approx)

        ##conv2
        d1 = conv_parallel(pool1_conv2M, bias_pool1_conv2, d1)
        d1 = activate_parallel(d1, relu_approx)

        ##conv3
        d1 = conv_parallel(pool1_conv3M, bias_pool1_conv3, d1)
        d1 = activate_parallel(d1, relu_approx)

        ##conv4
        d1 = conv_parallel(conv4M, bias_conv4M, d1)
        d1 = activate_parallel(d1, relu_approx)

        ##conv5
        d1 = conv_parallel(conv4M, bias_conv5M, d1)
        d1 = activate_parallel(d1, relu_approx)

        ##pool
        d1 = conv_parallel(pool1_pool2, None, d1)

        ##dense
        d1 = conv_parallel(dense1M, bias_dense1, d1)
        d1 = activate_parallel(d1, relu_approx)

        d1 = conv_parallel(dense2M, bias_dense2, d1)
        d1 = activate_parallel(d1, relu_approx)

        d1 = conv_parallel(dense3M, bias_dense3, d1)
        d1 = activate_parallel(d1, sigmoid_approx)

        print(d1.shape)

        out = torch_model(X_torch.double())
        _,pred = out.max(1)
        print("DIst")
        np.linalg.dist(d1-out.to_numpy().reshape(d1.shape))
        pred_l = np.argmax(d1,axis=1)
        correct_pytorch = correct_pytorch + (pred == Y).sum().item()
        correct_linear = correct_linear + np.sum(pred_l == Y.numpy())
        tot = tot + Y.shape[0]
    print(f"Accuracy torch {correct_pytorch/tot}")
    print(f"Accuracy linear {correct_linear/tot}")
    """