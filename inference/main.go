package main

import (
	"flag"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"os"
)

//Spawns an instance node into a server in the iccluster
//Suppose connection happens in LAN setting
const (
	usage = `usage: %s
//Spawns an instance node into a server in the iccluster
//Suppose connection happens in LAN setting

Options:
`
)

func main() {
	logN := flag.Int("logN", 14, "14 or 15 to choose params")
	model := flag.String("model", "crypto", "either 'crypto' for Cryptonets or 'nn' for ZAMA NN")
	nn := flag.Int("nn", 20, "20 or 50 to define model")
	addr := flag.String("addr", "127.0.0.1", "address of this node in cluster LAN")

	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), usage, os.Args[0])
		flag.PrintDefaults()
	}
	var params ckks.Parameters

	flag.Parse()

	if *model == "crypto" {
		if *logN == 14 {
			params = CNparamsLogN14Mask
		} else {
			panic("LogN15 is not supported for Cryptonet")
		}
	} else if *model == "nn" {
		if *logN == 14 {
			params = NNparamsLogN14
		} else if *logN == 15 {
			if *nn == 20 {
				params = NNparamsLogN15_NN20
			} else if *nn == 50 {
				params = NNparamsLogN15_NN50
			} else {
				panic("NN is only supported with 20 or 50 layers")
			}
		}
	} else {
		panic("Unknown model")
	}
	if *addr == "127.0.0.1" {
		panic("Please specify the address of this node")
	}

	//this creates a new player and listens for instructions from master
	instance := distributed.ListenForSetup(*addr, params)
	instance.Listen()
}
