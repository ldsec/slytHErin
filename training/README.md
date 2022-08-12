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

## Methods
### Convolution as Dense layer
Representation of ```gen_kernel_matrix```
![image](../inference/static/conv_trans.png)

Representation of ```pack_conv```
![image](../inference/static/conv.png)
