import onnx, onnx.helper as oh
import onnx.numpy_helper as nph
import argparse
import json

from training.conv_transform import serialize_nn

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
    ## to see the model
    #print(onnx.helper.printable_graph(onnx_model.graph))
    conv = nph.to_array(W[0])
    dense = [nph.to_array(w) for w in W[1:layers]]
    B = W[-layers:]
    bias_conv = nph.to_array(B[0])
    bias = [nph.to_array(b) for b in B[1:]]

    serialized = serialize_nn(conv,bias_conv,dense,bias,layers)

    with open(f'{j_name}', 'w') as f:
        json.dump(serialized, f)

