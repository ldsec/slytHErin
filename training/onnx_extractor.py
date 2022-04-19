import onnx, onnx.helper as oh
import onnx.numpy_helper as nph
import argparse
import json

"""to be used in docker container from ZAMA"""

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="onnx model: nn20, nn50, nn100",type=str)
    args = parser.parse_args()
  
    if args.model:
        if args.model == "nn20":
            layers = 21 ##1 conv + n dense
            path = "nn20_exe_onnx/fhe_model.onnx"
            j_name = "nn20.json"
        elif args.model == "nn50":
            layers = 51
            path = "nn50_exe_onnx/fhe_model.onnx"
            j_name = "nn50.json"
        elif args.model == "nn100":
            layers = 101
            path = "nn100_exe_onnx/fhe_model.onnx"
            j_name = "nn100.json"

    onnx_model = onnx.load(path)
    W = onnx_model.graph.initializer
    conv = nph.to_array(W[0])
    dense = [nph.to_array(w) for w in W[1:layers]]
    B = W[-layers:]
    bias_conv = nph.to_array(B[0])
    bias = [nph.to_array(b) for b in B[1:]]

    serialized = {}
    serialized['conv'] = {}
    serialized['conv']['weight'] = {'w': conv.tolist(),
    'kernels': conv.shape[0],
    'filters': conv.shape[1],
    'rows': conv.shape[2],
    'cols': conv.shape[3]}
    serialized['conv']['bias'] = {'b': bias_conv.tolist(), 'rows': 1, 'cols': bias_conv.shape[0]}
    for i in range(layers-1):
        serialized['dense_'+str(i+1)] = {}
        serialized['dense_'+str(i+1)]['weight'] = {'w': dense[i].tolist(), 'rows': dense[i].shape[0], 'cols': dense[i].shape[1]}
        serialized['dense_'+str(i+1)]['bias'] = {'b': bias[i].tolist(), 'rows': 1, 'cols': bias[i].shape[0]}

    with open(f'{j_name}', 'w') as f:
        json.dump(serialized, f)

