package main

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
)

//CRYPTONETS
var CNparamsLogN15, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ)
	LogP:         []int{50, 50, 50, 50},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     14,
	DefaultScale: float64(1 << 35),
})

var CNparamsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{60, 60},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

var CNparamsLogN14Mask, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{44, 44},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

//NN

var NNparamsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
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
var NNparamsLogN15_NN20, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

//Given a deg of approximation of 63 (so 6 level needed for evaluation) this set of params performs really good:
//It has 18 levels, so it invokes a bootstrap every 2 layers (1 lvl for mul + 6 lvl for activation) when the level
//is 4, which is the minimum level. In this case, bootstrap is called only when needed
//In case of NN50, cut the modulo chain at 11 levels, so to spare memory. In this case Btp happens every layer
var NNparamsLogN15_NN50, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

var NNparamsLogN16, _ = ckks.NewParametersFromLiteral(bootstrapping.N16QP1546H192H32.SchemeParams)
var NNbtpParamsLogN16 = bootstrapping.N16QP1546H192H32.BootstrappingParams
