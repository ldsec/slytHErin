
"""
    Script for testing nn with approximations for homomorphic encryption
"""
# explicit function to normalize array
def normalize(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def ReLU(X):
    relu = np.vectorize(lambda x: x * (x > 0))
    return relu(X)

def standard_eval(X,Y,serialized,layers):
    conv = np.array(serialized['conv']['weight']['w'])
    bias_conv = np.array(serialized['conv']['bias']['b'])
    dense, bias_dense = [],[]
    for i in range(layers):
        dense.append(np.array(serialized[f'dense_{str(i+1)}']['weight']['w']))
        bias_dense.append(np.array(serialized[f'dense_{str(i+1)}']['bias']['b']))
    
    CONV, CONV_BIAS = torch.from_numpy(conv).double(), torch.from_numpy(bias_conv).double()
    X = F.conv2d(X, CONV, CONV_BIAS, stride=1, padding=1)
    X = F.relu(X)
    X = X.reshape(X.shape[0], -1)
    iter = 0
    for d,b in zip(dense, bias_dense):
        D,B = torch.from_numpy(d).double(), torch.from_numpy(b).double()
        X = F.linear(X, D, B)
        if iter != layers-1:
            X = F.relu(X)
        iter += 1
    _, predicted_labels = X.max(1)
    corrects = (predicted_labels == Y).sum().item()

    return corrects

def linear_eval(X,Y, serialized,layers):
    """
        Linear pipeline without normalization and regular relu works fine
        Problem is when introducing relu_approx which need normalization to stay within interval
    """
    #conv = np.array(serialized['conv']['weight']['w'])
    #bias_conv = np.array(serialized['conv']['bias']['b'])
    #CONV, CONV_BIAS = torch.from_numpy(conv).double(), torch.from_numpy(bias_conv).double()
    #exp = F.conv2d(X, CONV, CONV_BIAS, stride=1, padding=1)
    
    X = F.pad(X, [1,1,1,1])
    X = X.reshape(X.shape[0],-1)
    
    conv, convMT = pack_conv_rect(normalize(np.array(serialized['conv']['weight']['w'])), 10,11,1,30,30)
    ## since value get replicated in the bias packed, normalize before doing that
    bias_conv = pack_bias(normalize(np.array(serialized['conv']['bias']['b'])), 2, 840//2)

    dense, bias_dense = [],[]
    for i in range(layers):
        dense.append(normalize(np.array(serialized[f'dense_{str(i+1)}']['weight']['w'])))
        bias_dense.append(normalize(np.array(serialized[f'dense_{str(i+1)}']['bias']['b'])))
    conv,conv_bias = convMT, np.array(bias_conv['b'])
    X = X @ conv
    for i in range(len(X)):
        X[i] += conv_bias
    X = relu_approx(X)
    
    iter = 0
    for d,b in zip(dense, bias_dense):
        X = X @ d.T
        for i in range(len(X)):
            X[i] = X[i] + b
        
        for x in X.flatten():
            if x > interval or x < -interval:
                print("Outside interval:", x)
        if iter != len(dense)-1:
            X = relu_approx(X)
        iter += 1

        #print("mean after activation:", X.mean()) #ok if normalized

    pred = np.argmax(X,axis=1)
    #print("Results: ", pred)
    #print("Expected labels: ", Y)
    corrects = np.sum(pred == Y.numpy())

    return corrects

def test_pipeline(eval):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="simplenet, nn20, nn50, nn100",type=str)
    args = parser.parse_args()
    
    with open(f'./models/{args.model}_packed.json', 'r') as f:
        serialized = json.load(f)
    if args.model == "nn20":
        layers = 20
    elif args.model == "nn50":
        layers = 50
    elif args.model == "nn100":
        layers = 100
    batchsize = 512
    dataHandler = DataHandler(dataset="MNIST", batch_size=batchsize)
    corrects = 0.0
    tot = 0
    for X,Y in dataHandler.test_dl:
        corrects += eval(X.double(),Y.double(),serialized,layers)
        tot += batchsize
    print("Accuracy:")
    print(corrects/tot)