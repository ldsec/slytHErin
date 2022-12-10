#!/bin/bash

# this scripts install go on the iccluster

#wget https://go.dev/dl/go1.18.3.linux-amd64.tar.gz
#tar -C /usr/local -xzf go1.18.3.linux-amd64.tar.gz
#
#echo "export GOROOT=/usr/local/go" >> ~/.bash_profile
#echo "export PATH=$PATH:$GOROOT/bin" >> ~/.bash_profile
#source .profile
apt install -y snapd
snap install go --classic
apt update -y
apt upgrade -y
apt install -y gcc