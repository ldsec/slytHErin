package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"math"
	"os"
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

func GenRotations(rowIn, colIn, numWeights int, rowsW, colsW []int, params ckks.Parameters, btpParams *bootstrapping.Parameters) []int {
	rotations := []int{}
	if btpParams != nil {
		rotations = btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
	}
	var replicationFactor int
	currCols := colIn
	for w := 0; w < numWeights; w++ {
		for i := 1; i < (rowsW[w]+1)>>1; i++ {
			rotations = append(rotations, 2*i*rowIn)
		}
		rotations = append(rotations, rowsW[w])
		rotations = append(rotations, -rowsW[w]*rowIn)
		rotations = append(rotations, -2*rowsW[w]*rowIn)
		if rowsW[w] < colsW[w] {
			replicationFactor = plainUtils.Max(int(math.Ceil(float64(colsW[w]/rowsW[w]))), 3)
			rotations = append(rotations, params.RotationsForReplicateLog(rowIn*currCols, replicationFactor)...)
		}
		currCols = colsW[w]
	}
	rotations = append(rotations, rowIn)
	return rotations
}

func SerializeKeys(path string, sk *rlwe.SecretKey, rotKeys *rlwe.RotationKeySet) {
	fmt.Println("Writing keys to disk: ", path)
	dat, err := sk.MarshalBinary()
	utils.ThrowErr(err)
	f, err := os.Create(path + "_sk")
	utils.ThrowErr(err)
	_, err = f.Write(dat)
	utils.ThrowErr(err)
	f.Close()

	dat, err = rotKeys.MarshalBinary()
	utils.ThrowErr(err)
	f, err = os.Create(path + "_rtks")
	utils.ThrowErr(err)
	_, err = f.Write(dat)
	utils.ThrowErr(err)
	f.Close()
}

func DesereliazeKeys(path string) (*rlwe.SecretKey, *rlwe.RotationKeySet) {
	fmt.Println("Reading keys from disk: ", path)
	dat, err := os.ReadFile(path + "_sk")
	utils.ThrowErr(err)
	var sk rlwe.SecretKey
	sk.UnmarshalBinary(dat)

	dat, err = os.ReadFile(path + "_rtks")
	utils.ThrowErr(err)
	var rotKeys rlwe.RotationKeySet
	rotKeys.UnmarshalBinary(dat)
	return &sk, &rotKeys
}
