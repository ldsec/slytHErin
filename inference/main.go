package main

import (
	"flag"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
)

var paramsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogSlots:     13,
	LogQ:         []int{40, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
	LogP:         []int{55},
	DefaultScale: 1 << 31,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

//Given a deg of approximation of 63 (so 6 level needed for evaluation) this set of params performs really good:
//It has 18 levels, so it invokes a bootstrap every 2 layers (1 lvl for mul + 6 lvl for activation) when the level
//is 4, which is the minimum level. In this case, bootstrap is called only when needed
//In case of NN50, cut the modulo chain at 11 levels, so to spare memory. In this case Btp happens every layer
var paramsLogN15_NN20, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})
var paramsLogN15_NN50, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

//Spawn a player node into a server in the iccluster
//Suppose connection happens in LAN setting
func main() {
	logN := flag.Int("logN", 14, "14 or 15 to choose params")
	nn := flag.Int("nn", 20, "20 or 50 to define model")
	var params ckks.Parameters

	flag.Parse()
	if *logN == 14 {
		params = paramsLogN14
	} else {
		if *nn == 20 {
			params = paramsLogN15_NN20
		} else {
			params = paramsLogN15_NN50
		}
	}

	setupAddr := "127.0.0.1:7000"
	//this creates a new player and listens for instructions from master
	distributed.Setup(setupAddr, params)
}
