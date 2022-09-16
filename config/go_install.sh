#!/bin/bash

# this scripts install go on the iccluster

wget https://go.dev/dl/go1.18.3.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.18.3.linux-amd64.tar.gz

export GOROOT=/usr/local/go
export PATH=$PATH:$GOROOT/bin

apt update -y
apt upgrade -y
apt install -y gcc