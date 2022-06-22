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
	sk           *rlwe.SecretKey
	kgen         ckks.KeyGenerator
}

func NewBox(params ckks.Parameters) CkksBox {
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	//init rotations
	//rotations are performed between submatrixes

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)

	Box := CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: kgen.GenRelinearizationKey(sk, 2)}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
		sk:           sk,
		kgen:         kgen,
	}
	return Box
}

func BoxShallowCopy(Box CkksBox) CkksBox {
	boxNew := CkksBox{
		Params:    Box.Params,
		Encoder:   Box.Encoder.ShallowCopy(),
		Evaluator: Box.Evaluator.ShallowCopy(),
		Decryptor: nil,
		Encryptor: nil,
	}
	return boxNew
}

//returns Box with Evaluator and Bootstrapper if needed
func BoxWithEvaluators(Box CkksBox, btpParams *bootstrapping.Parameters, withBtp bool, rowIn, colIn, numWeights int, rowsW, colsW []int) CkksBox {
	rotations := GenRotations(rowIn, colIn, numWeights, rowsW, colsW, Box.Params, btpParams)
	rlk := Box.kgen.GenRelinearizationKey(Box.sk, 2)
	rtks := Box.kgen.GenRotationKeysForRotations(rotations, true, Box.sk)
	Box.Evaluator = ckks.NewEvaluator(Box.Params, rlwe.EvaluationKey{
		Rlk:  rlk,
		Rtks: rtks,
	})
	var err error
	if withBtp {
		evk := bootstrapping.GenEvaluationKeys(*btpParams, Box.Params, Box.sk)
		Box.BootStrapper, err = bootstrapping.NewBootstrapper(Box.Params, *btpParams, evk)
		utils.ThrowErr(err)
	}
	return Box
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
