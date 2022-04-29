## Accuracy on 1000 samples: extracted NN model vs ZAMA original NN Model

This it the cleartext model.
The extracted model comes from ZAMA docker container
The extracted model is not 100% the same as ZAMA since we do not have enough information on how and which 
activation functions they use.
The results from ZAMA can be found in the docker container in /data/ext
| Model | Ours | Zama |
|-------|------|------|
| nn20  | 0.914 | 0.946 |
| nn50  | 0.95  | 0.913 |
| nn100 | 0.897 | 0.882 |