# slytHErin: An Agile Framework for Encrypted Deep Neural Network Inference
This repository contains the implementation code for "slytHErin: An Agile Framework for Encrypted Deep Neural Network Inference" that is accepted at  5th Workshop on Cloud Security and Privacy (Cloud S&P 2023). As the title suggests, "slytHErin" provides modular blocks
to easily build neural network which can be used for privacy-preserving inference under.
homomorphic encryption under different scenarios and threat model

< Insert reference >

## /training
This folder contains python scripts for training the models used for evaluation in the paper.
It contains serialization scripts for porting these models into a format which can be used by the inference part of the framework.
For more details refer to ```/training/README.md```

## /inference
This folder contains the main framework written in Go for privacy-preserving inference using homomorphic encryption.
For more details refer to ```/inference/README.md```


