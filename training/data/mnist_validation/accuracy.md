## Accuracy on 1000 samples: extracted NN model vs ZAMA original NN Model

This it the cleartext model. The extracted model comes from ZAMA docker container
The extracted model is not 100% the same as ZAMA since we do not have enough information on how and which 
activation functions they use.

| Model | Ours | Zama |
|-------|------|------|
| nn20  | 0.914 | 97.5 |
| nn50  | 0.95  | 95.4 |
| nn100 | 0.897 | 95.2 |