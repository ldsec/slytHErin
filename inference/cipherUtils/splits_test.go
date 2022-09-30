package cipherUtils

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"testing"
)

func TestFindSplits_CryptoNet(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
		LogP:         []int{42, 42},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 30),
	})
	S := NewSplitter(false, -1, 28*28, []int{784, 720, 100}, []int{720, 100, 10}, params)
	split := S.FindSplits()
	split.Print()
}

func TestFindSplits_NN(t *testing.T) {
	inputFeatures := 784
	layers := 20
	weightRows := make([]int, layers+1)
	weightCols := make([]int, layers+1)
	weightRows[0] = 784
	weightCols[0] = 676
	weightRows[1] = 676
	weightCols[1] = 92
	for i := 2; i < layers+1; i++ {
		weightRows[i] = 92
		weightCols[i] = 92
	}
	weightCols[layers] = 10

	ckksParams := ckks.ParametersLiteral{
		LogN:         14,
		LogSlots:     13,
		LogQ:         []int{40, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
		LogP:         []int{55},
		DefaultScale: 1 << 31,
		Sigma:        rlwe.DefaultSigma,
		RingType:     ring.Standard,
	}
	params, _ := ckks.NewParametersFromLiteral(ckksParams)
	S := NewSplitter(true, -1, inputFeatures, weightRows, weightCols, params)
	split := S.FindSplits()
	split.Print()

}
