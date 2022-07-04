package multidim

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	bootstrapping2 "github.com/ldsec/lattigo/v2/ckks/bootstrapping"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"math"
	"os"
)

//Same as box, but for ckks v2
type Ckks2Box struct {
	//wrapper for the classes needed to perform encrypted operations, like a crypto-ToolBox
	Params       ckks2.Parameters
	Encoder      ckks2.Encoder
	Evaluator    ckks2.Evaluator
	Encryptor    ckks2.Encryptor
	Decryptor    ckks2.Decryptor
	BootStrapper *bootstrapping2.Bootstrapper
	PoolIdx      int //id for threads
}

func GenRotations(rowIn, colIn, numWeights int, rowsW, colsW []int, params ckks2.Parameters, btpParams *bootstrapping.Parameters) []int {
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

func SerializeKeys(path string, sk *rlwe2.SecretKey, rotKeys *rlwe2.RotationKeySet) {
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

func DesereliazeKeys(path string) (*rlwe2.SecretKey, *rlwe2.RotationKeySet) {
	fmt.Println("Reading keys from disk: ", path)
	dat, err := os.ReadFile(path + "_sk")
	utils.ThrowErr(err)
	var sk rlwe2.SecretKey
	sk.UnmarshalBinary(dat)

	dat, err = os.ReadFile(path + "_rtks")
	utils.ThrowErr(err)
	var rotKeys rlwe2.RotationKeySet
	rotKeys.UnmarshalBinary(dat)
	return &sk, &rotKeys
}
