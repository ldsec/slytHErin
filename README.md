# GeF : A Generic Framework for Secure Deep Neural Network Inference
This repository contains the implementation code for "Gef : A Generic Framework for Secure Neural Network Inference". As the title suggests, "Gef" provides modular blocks
to easily build neural network which can be used for privacy-preserving inference under.
homomorphic encryption under different scenarios and threat model

< Insert reference >

## /training
This folder contains python scripts for training the Cryptonet model used for evaluation in the paper. It contains also ```.json``` files with the parameters of NN models[1] already trained using a proprietary library.
More importantly, it contains serialization scripts for porting these models into a format which can be used by the inference part of the framework.
For more details refer to ```/training/README.md```

## /inference
This folder contains the main framework written in Go for privacy-preserving inference using homomorphic encryption.
For more details refer to ```/inference/README.md```

[1] I. Chillotti, M. Joye, and P. Paillier. Programmable bootstrapping enables efficient homomorphic inference of deep neural networks. Cryptology ePrint Archive, Paper 2021/091, 2021. https://eprint.iacr.org/2021/091.
