# What is this?
We use a remote cluster to test the remote scenarios for Cryptonets and for ZAMA NN networks

## How to setup remote cluster environment
- 0) First be sure to build the executable in the ```inference/``` directory
with ```go build``` and that you have ```config.json``` in this folder

You can skip steps 1 and 2 if you already have populated the config file with the ```cluster_ips```
- 1) First populate the config.json with ssh credentials, the number of cluster machines and their id (only the numbers, e.g 45 for iccluster045)
- 2) Run ```ip_scan.sh```
- 3) Run ```setup.sh```
- 4) When you are ready, run ```remote.sh``` to run the instances
on the remote servers.
- 5) When you are done with the experiments, kill the processes with ```cleanup.sh```