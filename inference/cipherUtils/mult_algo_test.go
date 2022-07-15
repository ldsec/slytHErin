package cipherUtils

import (
	"fmt"
	pU "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"gonum.org/v1/gonum/mat"
	"testing"
	"time"
)

/********************************************
EXPERIMENTS ON MULTIPLICATION ALGORITHM
*********************************************/

func TestMultiplicationAlgoC2P(t *testing.T) {
	rowx := 4
	colx := 4
	roww := 4
	colw := 4
	//X := pU.RandMatrix(rowx, colx)
	//W := pU.RandMatrix(roww, colw)
	Xv := make([][]float64, rowx)
	for i := 0; i < rowx; i++ {
		Xv[i] = make([]float64, colx)
		for j := 0; j < colx; j++ {
			Xv[i][j] = 1.0 + float64(i*colx+j)*1e-0
		}
	}
	X := pU.NewDense(Xv)
	Wv := make([][]float64, roww)
	for i := 0; i < roww; i++ {
		Wv[i] = make([]float64, colw)
		for j := 0; j < colw; j++ {
			Wv[i][j] = 1.0 + float64(i*colw+j)*1e-0
			fmt.Println(Wv[i][j])
		}
		fmt.Println()
	}
	W := pU.NewDense(Wv)

	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	Box := NewBox(params)
	Box = BoxWithRotations(Box, GenRotations(pU.NumRows(X), pU.NumCols(X), 1, []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, []int{}, []int{}, params, nil), false, bootstrapping.Parameters{})

	t.Run("Test/C2P/Test", func(t *testing.T) {
		input := EncryptInput(params.MaxLevel(), params.DefaultScale(), pU.MatToArray((X)), Box)
		Wd, _ := FormatWeightsAsMap(pU.MatToArray(W), pU.NumRows(X), false)
		Wlt := ckks.GenLinearTransformBSGS(Box.Encoder, Wd, params.MaxLevel(), params.QiFloat64(input.Level()), 2.0, params.LogSlots())
		start := time.Now()
		rotations := GenRotations(pU.NumRows(X), pU.NumCols(X), 1, []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, []int{}, []int{}, params, &bootstrapping.Parameters{})
		rotations = append(rotations, Wlt.Rotations()...)
		Box = BoxWithRotations(Box, rotations, false, bootstrapping.Parameters{})

		res := Box.Evaluator.LinearTransformNew(input, Wlt)[0]

		done := time.Since(start)
		var resPlain mat.Dense
		resPlain.Mul(pU.TransposeDense(W), X)

		//we need to tranpose the plaintext result according to the diagonalized multiplication algo
		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(res, valuesWant, 0.001, Box)
		fmt.Println("Done ", done)
	})
	t.Run("Test/C2P", func(t *testing.T) {
		Xenc := EncryptInput(params.MaxLevel(), params.DefaultScale(), pU.MatToArray(X), Box)
		Wenc := EncodeWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), Box)
		ops := make([]ckks.Operand, len(Wenc))
		for i := range Wenc {
			ops[i] = Wenc[i]
		}
		start := time.Now()
		Renc := DiagMul(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), ops, true, true, Box)
		done := time.Since(start)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		//we need to tranpose the plaintext result according to the diagonalized multiplication algo
		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
		fmt.Println("Done ", done)
	})

}
