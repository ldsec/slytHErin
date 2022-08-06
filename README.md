# dnn-inference
Secure, private, efficient DNN inference

## Training
This folder contains python and go scripts for training cryptonet model and nn models, as well as serialization scripts for porting these models into the inference part of the code.

## Cryptonet
You can run ```python3 cryptonet.py``` to train a new cryptonet model.
After this you should see a ```cryptonet_packed.json``` file in the ```models```
folder. You can copy this to ```inference/cryptonet```

### Data
You can generate the json with MNIST data with ```python3 dataHandler.py --model cryptonet```
and find the file in the ```data``` folder as a json

## NN
You are provided with the json files already generated

### Data
Same as cryptonet, use dataHandler with ```--model nn```