package modelsPlain

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
)

type SimpleNet struct {
	conv1 [][]float64
	pool1 [][]float64
	pool2 [][]float64

	reluApprox ckks.Polynomial //this will store the coefficients of the poly approximating ReLU
}
