package cipherUtils

import (
	"errors"
	"fmt"
	pU "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"gonum.org/v1/gonum/mat"
	"runtime"
	"testing"
	"time"
)

func TestMultiplier_Multiply(t *testing.T) {
	X := pU.RandMatrix(64, 64)
	W := pU.RandMatrix(64, 32)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	splits := FindSplits(pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params, false)
	if len(splits) == 0 {
		panic(errors.New("No splits found"))
	}

	Box := NewBox(params)

	t.Run("Test/C2P", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), Box)
		Wpt, _ := NewPlainWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, Xenc.InnerRows, Xenc.InnerCols, 1, []int{Wpt.InnerRows}, []int{Wpt.InnerCols})
		Mul := NewMultiplier(Box, 1)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})

	t.Run("Test/C2P/Multithread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")

		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), Box)
		Wpt, _ := NewPlainWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, Xenc.InnerRows, Xenc.InnerCols, 1, []int{Wpt.InnerRows}, []int{Wpt.InnerCols})
		Mul := NewMultiplier(Box, runtime.NumCPU())
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})

	t.Run("Test/C2C", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), Box)
		Wct, _ := NewEncWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, Xenc.InnerRows, Xenc.InnerCols, 1, []int{Wct.InnerRows}, []int{Wct.InnerCols})
		Mul := NewMultiplier(Box, 1)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wct)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})

	t.Run("Test/C2C/Multithread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")

		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), Box)
		Wct, _ := NewEncWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, Xenc.InnerRows, Xenc.InnerCols, 1, []int{Wct.InnerRows}, []int{Wct.InnerCols})
		Mul := NewMultiplier(Box, runtime.NumCPU())
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wct)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})
}

func TestAdder_AddBias(t *testing.T) {
	X := pU.RandMatrix(64, 64)
	B := pU.RandMatrix(64, 64)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	Box := NewBox(params)

	t.Run("Test/Add", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), Box)
		Bpt, _ := NewPlainInput(B, 4, 4, params.MaxLevel(), Box)
		Ad := NewAdder(Box, 1)
		start := time.Now()
		Ad.AddBias(Xenc, Bpt)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Add(X, B)
		resPt, _ := pU.PartitionMatrix(&res, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)

	})
	t.Run("Test/Add/Multitrehad", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")
		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), Box)
		Bpt, _ := NewPlainInput(B, 4, 4, params.MaxLevel(), Box)
		Ad := NewAdder(Box, runtime.NumCPU())
		start := time.Now()
		Ad.AddBias(Xenc, Bpt)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Add(X, B)
		resPt, _ := pU.PartitionMatrix(&res, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
}

func TestActivator_ActivateBlocks(t *testing.T) {
	X := pU.RandMatrix(64, 64)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	Box := NewBox(params)

	activation := utils.InitReLU(3)
	t.Run("Test/Activate", func(t *testing.T) {
		// we need to rescale the input before the activation
		Xscaled := pU.MulByConst(pU.NewDense(pU.MatToArray(X)), 1.0/activation.Interval)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), Box)
		Act, _ := NewActivator(activation, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box, 1)

		start := time.Now()
		Act.ActivateBlocks(Xenc)
		fmt.Println("Done: ", time.Since(start))

		Xc := pU.NewDense(pU.MatToArray(X))
		utils.ActivatePlain(Xc, activation) //this automatically rescales the input before activating

		resPt, _ := pU.PartitionMatrix(Xc, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
	t.Run("Test/Activate/MultiThread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")
		Xscaled := pU.MulByConst(pU.NewDense(pU.MatToArray(X)), 1.0/activation.Interval)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), Box)
		Act, _ := NewActivator(activation, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box, runtime.NumCPU())

		start := time.Now()
		Act.ActivateBlocks(Xenc)
		fmt.Println("Done: ", time.Since(start))

		Xc := pU.NewDense(pU.MatToArray(X))
		utils.ActivatePlain(Xc, activation)

		resPt, _ := pU.PartitionMatrix(Xc, Xenc.RowP, Xenc.ColP)

		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
}
