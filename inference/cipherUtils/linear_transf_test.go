package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"testing"
)

var CNparamsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{45, 40, 40}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{60, 60},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 40),
})

//Assume W is square
func FormatWeightMap(W [][]float64) (map[int][]float64, error) {
	if len(W) != len(W[0]) {
		return nil, errors.New("Non square")
	}
	d := len(W)
	nonZeroDiags := make(map[int][]float64)
	for i := 0; i < d; i++ {
		isZero := true
		diag := make([]float64, d*2)
		for j := 0; j < d; j++ {
			diag[j] = W[(j)%d][(j+i)%d]
			if diag[j] != 0 {
				isZero = false
			}
		}
		if !isZero {
			nonZeroDiags[i] = plainUtils.ReplicateRealArray(diag, 2)
		}
	}
	return nonZeroDiags, nil
}

func Test_LinearTransfor(t *testing.T) {
	r := 3
	c := 3
	A := plainUtils.RandMatrix(c, 1)
	W := plainUtils.RandMatrix(r, c)
	plainUtils.PrintDense(A)
	plainUtils.PrintDense(W)

	R := make([]float64, c)

	Box := NewBox(CNparamsLogN14)

	ctA := Box.Encryptor.EncryptNew(Box.Encoder.EncodeNew(plainUtils.RowFlatten(A), CNparamsLogN14.MaxLevel(), CNparamsLogN14.DefaultScale(), CNparamsLogN14.LogSlots()))
	diagW, err := FormatWeightMap(plainUtils.MatToArray(W))
	utils.ThrowErr(err)
	lt := ckks.GenLinearTransformBSGS(Box.Encoder, diagW, CNparamsLogN14.MaxLevel(), CNparamsLogN14.QiFloat64(CNparamsLogN14.MaxLevel()), 8, CNparamsLogN14.LogSlots())
	rotations := CNparamsLogN14.RotationsForReplicateLog(c, 2)
	rotations = append(rotations, lt.Rotations()...)

	Box = BoxWithRotations(Box, rotations, false, bootstrapping.Parameters{})
	Box.Evaluator.ReplicateLog(ctA, c*1, 2, ctA)
	ctR := Box.Evaluator.LinearTransformNew(ctA, lt)[0]
	ReImg := Box.Encoder.Decode(Box.Decryptor.DecryptNew(ctR), CNparamsLogN14.LogSlots())[:(r * 1)]
	Re := make([]float64, len(ReImg))
	for i := range ReImg {
		Re[i] = real(ReImg[i])
	}

	fmt.Println("Want")
	for i := range diagW {
		rot := plainUtils.RotateRealArray(plainUtils.RowFlatten(A), -i)
		for j := range R {
			R[j] += rot[j] * diagW[i][j]
		}
	}
	fmt.Println(R)
	fmt.Println("Really want")
	M := new(mat.Dense)
	M.Mul(W, A)
	fmt.Println(plainUtils.RowFlatten(M))
	fmt.Println("Get")
	fmt.Println(Re)
}
