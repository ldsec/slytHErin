package cipherUtils

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
)

type CkksBox struct {
	//wrapper for the classes needed to perform encrypted operations, like a crypto-ToolBox
	Params       ckks.Parameters
	Encoder      ckks.Encoder
	Evaluator    ckks.Evaluator
	Encryptor    ckks.Encryptor
	Decryptor    ckks.Decryptor
	BootStrapper *bootstrapping.Bootstrapper
}

func GenRotations(dimIn, numWeights int, rowsW, colsW []int, params ckks.Parameters, btpParams *bootstrapping.Parameters) []int {
	rotations := []int{}
	if btpParams != nil {
		rotations = btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
	}
	for w := 0; w < numWeights; w++ {
		for i := 1; i < (rowsW[w]+1)>>1; i++ {
			rotations = append(rotations, 2*i*dimIn)
		}
		rotations = append(rotations, rowsW[w])
		rotations = append(rotations, -rowsW[w]*dimIn)
		rotations = append(rotations, params.RotationsForReplicateLog(dimIn*rowsW[w], 3)...)
		rotations = append(rotations, params.RotationsForReplicateLog(dimIn*colsW[w], 3)...)
	}
	rotations = append(rotations, dimIn)
	return rotations
}
