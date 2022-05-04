## Accuracy on 1000 samples: extracted NN model vs ZAMA original NN Model

- The extracted model comes from ZAMA docker container
- Activation is relu for each layer but last one where it is identity
- The results from ZAMA can be found in the docker container in /data/ext --> those are for inference on encrypted data
- Zama Clear is unencrypted inference from their paper

| Model | Zama Clear | Zama Enc |
|-------|------------|----------|
| nn20  | 0.975      | 0.946    |
| nn50  | 0.954      | 0.913    |
| nn100 | 0.952      | 0.882    |

- This is everything in clear

| Model | Ours       |
|-------|------------|
| nn20  | 0.914      |
| nn50  | 0.950      |
| nn100 | 0.897      |

