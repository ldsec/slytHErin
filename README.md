# dnn-inference
Secure, private, efficient DNN inference

## Updates

### Go Side
Fully implemented:
-   Inference on CryptoNet using encrypted data, cleartext model
-   Inference on **untrained** NN20 with both data and model encrypted, with centralized bootstrap

Partially implemented:
-   Inference on NN20 in distributed setting. Currently there is a PoC using channels and go routine.

Next steps:
-   Implement a version using TCP sockets (is TLS even needed?) using localhost interface

### Python side
Fully implented:
-   CryptoNet

Partially implemented:
-   NN training works using Adam + Cross Entropy + relu (or anyway no approximated activation)

Next steps:
-   We should solve the issue when running inference of NN where we have values outside the approximation interval. Possible solutions:
    -   Clipping
    -   Training with approximated activation --> really slow convergence using Adam and Cross Entropy