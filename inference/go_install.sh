#!/bin/sh

# this scripts install go on the iccluster
export GOROOT=/usr/local/go
export PATH=$PATH:$GOROOT/bin

get -q -O - https://raw.githubusercontent.com/canha/golang-tools-install-script/master/goinstall.sh | bash
source /root/.bashrc
