package cipherUtils

import (
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"math/rand"
	"testing"
)

/********************************************
BLOCK MATRICES OPS
|
|
v
*********************************************/

func C2PTest(t *testing.T) {
	LDim := []int{96, 64}
	W0Dim := []int{64, 64}
	W1Dim := []int{64, 64}

	r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = r.NormFloat64()
		}
	}

	W0 := make([][]float64, W0Dim[0])
	for i := range W0 {
		W0[i] = make([]float64, W0Dim[1])

		for j := range W0[i] {
			W0[i][j] = r.NormFloat64()
		}
	}

	W1 := make([][]float64, W1Dim[0])
	for i := range W1 {
		W1[i] = make([]float64, W1Dim[1])

		for j := range W1[i] {
			W1[i][j] = r.NormFloat64()
		}
	}

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 6, 8)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 8, 4)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 4, 2)

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40},
		LogP:         []int{61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	if err != nil {
		panic(err)
	}

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := []int{}
	for i := 1; i < W0b.InnerRows; i++ {
		rotations = append(rotations, 2*i*Lb.InnerRows)
	}

	for i := 1; i < W1b.InnerRows; i++ {
		rotations = append(rotations, 2*i*Lb.InnerRows)
	}

	rotations = append(rotations, Lb.InnerRows)
	rotations = append(rotations, W0b.InnerRows)
	rotations = append(rotations, W1b.InnerRows)
	rotations = append(rotations, -W0b.InnerRows*Lb.InnerRows)
	rotations = append(rotations, -2*W0b.InnerRows*Lb.InnerRows)
	rotations = append(rotations, -W1b.InnerRows*Lb.InnerRows)
	rotations = append(rotations, -2*W1b.InnerRows*Lb.InnerRows)

	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	Box := CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}

	ctA, err := NewEncInput(L, 6, 8, Box)
	W0bp, err := NewPlainWeight(W0, 8, 4, ctA.InnerRows, Box)
	W1bp, err := NewPlainWeight(W0, 4, 2, ctA.InnerRows, Box)

}
