## CryptoNets training problem
I managed to identify the problem, with a certain degree of certainty, to the use of the square activation function that will "blow up" the inputs. Probably what happens at that point is that the Sigmoid computed in the backprop. step becames 0...
Putting ReLU in place of the square seems to solve the problem, even if the architecture is weird with the sum pooling.
Also initializing weights after the square to small values seems to be good but not enough

## SimpleNet
In the paper for CryptoNet they mention a simplified 5-layer network used for inference. Apparently this one works much (much!)better during the training.
Many combinations of activation function (square, relu, polynomial approx of relu up to degree=4) have been experimented. Best seems to be xavier init + relu or square.

## AlexNet
AlexNet with Sum Pooling instead of MaxPooling...waiting for training. Architecture is been implemented following the PyTorch standard one