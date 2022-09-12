import torch
from packing import *
import numpy as np

def test_conv():
    X = torch.randn(2,3,28,28)
    M = torch.randn(1,3,4,2)

    R = torch.nn.functional.conv2d(X,M,padding=0, stride=2)
    Rf = flat_tensor(R)

    _, T = pack_conv(M.detach().numpy(),4,2,2,28,28)
    x = flat_tensor(X.detach().numpy())

    print("Flattened x has shape: ", x.shape)
    print("Flattened and transformed W has shape: ", T.shape)
    rf = x @ T
    for a,b in zip(Rf.flatten(), rf.flatten()):
        assert abs(a-b) < 1**(-10)

if __name__ == "__main__":
    test_conv()

