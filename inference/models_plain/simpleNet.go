package models_plain

import (
	"github.com/ldsec/lattigo/v2/ckks"
	mats "github.com/ldsec/new_poseidon/matrices"
)

type SimpleNet struct {
	conv1 mats.PlaintextBatchMatrix
	pool1 mats.PlaintextBatchMatrix
	pool2 mats.PlaintextBatchMatrix

	reluApprox ckks.Polynomial //this will store the coefficients of the poly approximating ReLU
}
