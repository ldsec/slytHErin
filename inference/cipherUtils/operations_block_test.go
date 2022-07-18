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
	X := pU.RandMatrix(40, 720)
	W := pU.RandMatrix(720, 100)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	splits := FindSplits(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
	PrintAllSplits(splits)
	if len(splits) == 0 {
		panic(errors.New("No splits found"))
	}

	Box := NewBox(params)

	t.Run("Test/C2P", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewPlainWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits[0])
		Mul := NewMultiplier(Box, 1)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})

	t.Run("Test/C2P/Multithread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")

		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewPlainWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits[0])
		Mul := NewMultiplier(Box, runtime.NumCPU())
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})

	t.Run("Test/C2C", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), params.DefaultScale(), Box)
		Wct, _ := NewEncWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits[0])
		Mul := NewMultiplier(Box, 1)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wct, true)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})

	t.Run("Test/C2C/Multithread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")

		Xenc, _ := NewEncInput(X, splits[0][0].RowP, splits[0][0].ColP, params.MaxLevel(), params.DefaultScale(), Box)
		Wct, _ := NewEncWeightDiag(W, splits[0][1].RowP, splits[0][1].ColP, Xenc.InnerRows, params.MaxLevel(), Box)
		Box = BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits[0])
		Mul := NewMultiplier(Box, runtime.NumCPU())
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wct, true)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.01, Box)
	})
}

func TestAdder_AddBias(t *testing.T) {
	X := pU.RandMatrix(63, 64)
	B := pU.RandMatrix(63, 64)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	Box := NewBox(params)

	t.Run("Test/Add", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Bpt, _ := NewPlainInput(B, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
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
		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Bpt, _ := NewPlainInput(B, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
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

	t.Run("Test/Activate", func(t *testing.T) {
		activation := utils.InitReLU(3)
		// we need to rescale the input before the activation
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(activation, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box, 1)

		start := time.Now()
		Act.ActivateBlocks(Xenc)
		fmt.Println("Done: ", time.Since(start))

		utils.ActivatePlain(Xscaled, activation) //this automatically rescales the input before activating

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})

	t.Run("Test/Activate/MultiThread", func(t *testing.T) {
		activation := utils.InitReLU(3)
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(activation, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box, runtime.NumCPU())

		start := time.Now()
		Act.ActivateBlocks(Xenc)
		fmt.Println("Done: ", time.Since(start))

		utils.ActivatePlain(Xscaled, activation)

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)

		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})

	t.Run("Test/Activate/Chebybase", func(t *testing.T) {
		activation := utils.InitActivationCheby("silu", -5, 5, 10)
		// we need to rescale the input before the activation
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(activation, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box, 1)

		start := time.Now()
		Act.ActivateBlocks(Xenc)
		fmt.Println("Done: ", time.Since(start))

		utils.ActivatePlain(Xscaled, activation)

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})

	t.Run("Test/Activate/MultiThread/Chebybase", func(t *testing.T) {
		activation := utils.InitActivationCheby("silu", -5, 5, 10)
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(activation, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box, runtime.NumCPU())

		start := time.Now()
		Act.ActivateBlocks(Xenc)
		fmt.Println("Done: ", time.Since(start))

		utils.ActivatePlain(Xscaled, activation)

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)

		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
}

func TestBootstrapper_Bootstrap(t *testing.T) {
	X := pU.RandMatrix(64, 64)
	ckksParams := bootstrapping.N16QP1546H192H32.SchemeParams
	btpParams := bootstrapping.N16QP1546H192H32.BootstrappingParams

	params, _ := ckks.NewParametersFromLiteral(ckksParams)

	Box := NewBox(params)
	Box = BoxWithRotations(Box, GenRotations(16, 16, 0, []int{}, []int{}, []int{}, []int{}, params, &btpParams), true, btpParams)

	t.Run("Test/Bootstrap", func(t *testing.T) {

		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Btp := NewBootstrapper(Box, 1)
		start := time.Now()
		Btp.Bootstrap(Xenc)
		fmt.Println("Done: ", time.Since(start))

		Xc := pU.NewDense(pU.MatToArray(X))

		resPt, _ := pU.PartitionMatrix(Xc, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
	t.Run("Test/Bootstrap/MultiThread", func(t *testing.T) {

		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Btp := NewBootstrapper(Box, runtime.NumCPU())
		start := time.Now()
		Btp.Bootstrap(Xenc)
		fmt.Println("Done: ", time.Since(start))

		Xc := pU.NewDense(pU.MatToArray(X))

		resPt, _ := pU.PartitionMatrix(Xc, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
}

func TestRepack(t *testing.T) {
	rows := 1
	cols := 720
	rowP := 1
	colP := 10
	newColP := 8
	X := pU.RandMatrix(rows, cols)
	W := pU.RandMatrix(cols, 100)
	pU.PrintDense(X)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	Box := NewBox(params)
	rotations := GenRotationsForRepackCols(rows/rowP, cols, cols/colP, newColP)
	Box = BoxWithRotations(Box, rotations, false, bootstrapping.Parameters{})

	Xenc, err := NewEncInput(X, rowP, colP, params.MaxLevel(), params.DefaultScale(), Box)
	utils.ThrowErr(err)
	start := time.Now()
	RepackCols(Xenc, newColP, Box)
	done := time.Since(start)

	repack, _ := pU.PartitionMatrix(X, rowP, newColP)
	pU.PrintBlocks(repack)
	PrintDebugBlocks(Xenc, repack, 0.0001, Box)

	fmt.Println("Done repack: ", done)

	Wpt, err := NewPlainWeightDiag(W, newColP, 2, Xenc.InnerRows, params.MaxLevel()-1, Box)
	utils.ThrowErr(err)

	Box = BoxWithRotations(Box, GenRotations(Xenc.InnerRows, Xenc.InnerCols, 1, []int{Wpt.InnerRows}, []int{Wpt.InnerCols}, []int{Wpt.ColP}, []int{Wpt.RowP}, params, nil), false, bootstrapping.Parameters{})
	Mul := NewMultiplier(Box, runtime.NumCPU())
	start = time.Now()
	resEnc := Mul.Multiply(Xenc, Wpt, true)
	fmt.Println("Done: ", time.Since(start))

	var res mat.Dense
	res.Mul(X, W)
	resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
	PrintDebugBlocks(resEnc, resPt, 0.01, Box)
}

//helpers
/*
//Repack version with repacking also of rows (needs masking)
func Repack(X *EncInput, rowP, colP int, eval ckks.Evaluator) *EncInput {
	rows := X.RowP * X.InnerRows
	cols := X.ColP * X.InnerCols
	if rows%rowP != 0 || cols%colP != 0 || X.RowP%rowP != 0 || X.ColP%colP != 0 {
		panic(errors.New("Target Partition not compatible with given Block Matrix"))
	}
	if X.RowP*X.ColP == 1 {
		fmt.Println("Repacking: Nothing to do")
		return X
	}

	Xnew := &EncInput{
		Blocks:    nil,
		RowP:      rowP,
		ColP:      colP,
		InnerRows: rows / rowP,
		InnerCols: cols / colP,
	}

	innerBlocks := X.RowP / rowP
	buffer := make([][]*ckks.Ciphertext, rowP)
	if innerBlocks != 1 {
		for i := 0; i < X.ColP; i++ {
			//for each col partition
			for part := 0; part < rowP; part++ {
				//for each block in this part of this colP, extract the columns
				buffer[part] = make([]*ckks.Ciphertext, X.ColP)
				colsExt := make([]*ckks.Ciphertext, X.InnerCols)
				for j := part * innerBlocks; j < part*rowP+innerBlocks; j++ {
					colsExt[j] = ExtractCols(X.Blocks[j][i], X.InnerRows, X.InnerCols)
				}

				//collate the columns of each block of this part
				accumulator := X.Blocks[part*innerBlocks][i]
				for col := 0; col < X.InnerCols; col++ {
					for j := part * innerBlocks; j < part*rowP+innerBlocks; j++ {
						//accumulator += cols[j][col]
					}
				}
				buffer[part][i] = accumulator
			}
		}
	}
	innerBlocks = X.ColP / colP
	if innerBlocks != 1 {
		for i := 0; i < rowP; i++ {
			//for each row, unite blocks
			for part := 0; part < colP; part++ {
				accumulator := buffer[i][part*innerBlocks]
				for j := part*innerBlocks + 1; j < part*rowP+innerBlocks; j++ {
					eval.Add(accumulator, eval.RotateNew(X.Blocks[i][j], -X.InnerRows*cols), accumulator)
				}
			}
		}
	}

}
*/
