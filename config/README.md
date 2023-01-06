This folder contains all the json data needed to run the experiments
on a remote machine. It contains also the binary ```inference```
which spawns a remote service. This service can either represent a remote ```player``` for the experiments
involving distributed bootstrapping or a remote ```server``` that offers
an oblivious decryption oracle for the experiment involving offloading the
encrypted model to the ```client```.