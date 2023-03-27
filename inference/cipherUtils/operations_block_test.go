package cipherUtils

import (
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
	X := pU.RandMatrix(128, 128)
	W := pU.RandMatrix(128, 128)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	Box := NewBox(params)

	t.Run("Test/C2P", func(t *testing.T) {
		S := NewSplitter(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
		splits := S.FindSplits()
		splits.Print()
		Xenc, _ := NewEncInput(X, splits.ExctractInfoAt(0)[2], splits.ExctractInfoAt(0)[3], params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewPlainWeightDiag(W, splits.ExctractInfoAt(1)[2], splits.ExctractInfoAt(1)[3], Xenc.InnerRows, Xenc.InnerCols, params.MaxLevel(), Box)
		Box = BoxWithRotations(Box, Wpt.GetRotations(params), false, nil)
		Mul := NewMultiplier(1)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true, Box)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.001, Box)
	})

	t.Run("Test/C2P_Multithread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")

		S := NewSplitter(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
		splits := S.FindSplits()
		splits.Print()
		Xenc, _ := NewEncInput(X, splits.ExctractInfoAt(0)[2], splits.ExctractInfoAt(0)[3], params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewPlainWeightDiag(W, splits.ExctractInfoAt(1)[2], splits.ExctractInfoAt(1)[3], Xenc.InnerRows, Xenc.InnerCols, params.MaxLevel(), Box)
		Box = BoxWithRotations(Box, Wpt.GetRotations(params), false, nil)
		Mul := NewMultiplier(runtime.NumCPU())
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true, Box)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.001, Box)
	})

	t.Run("Test/C2C", func(t *testing.T) {
		S := NewSplitter(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
		splits := S.FindSplits()
		splits.Print()
		Xenc, _ := NewEncInput(X, splits.ExctractInfoAt(0)[2], splits.ExctractInfoAt(0)[3], params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewEncWeightDiag(W, splits.ExctractInfoAt(1)[2], splits.ExctractInfoAt(1)[3], Xenc.InnerRows, Xenc.InnerCols, params.MaxLevel(), Box)
		Box = BoxWithRotations(Box, Wpt.GetRotations(params), false, nil)
		Mul := NewMultiplier(1)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true, Box)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.001, Box)
	})

	t.Run("Test/C2C_Multithread", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")

		S := NewSplitter(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
		splits := S.FindSplits()
		splits.Print()
		Xenc, _ := NewEncInput(X, splits.ExctractInfoAt(0)[2], splits.ExctractInfoAt(0)[3], params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewEncWeightDiag(W, splits.ExctractInfoAt(1)[2], splits.ExctractInfoAt(1)[3], Xenc.InnerRows, Xenc.InnerCols, params.MaxLevel(), Box)
		Box = BoxWithRotations(Box, Wpt.GetRotations(params), false, nil)
		Mul := NewMultiplier(runtime.NumCPU())
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true, Box)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.001, Box)
	})

	t.Run("Test/P2C", func(t *testing.T) {
		S := NewSplitter(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
		splits := S.FindSplits()
		splits.Print()
		Xenc, _ := NewPlainInput(X, splits.ExctractInfoAt(0)[2], splits.ExctractInfoAt(0)[3], params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewEncWeightDiag(W, splits.ExctractInfoAt(1)[2], splits.ExctractInfoAt(1)[3], Xenc.InnerRows, Xenc.InnerCols, params.MaxLevel(), Box)
		Box = BoxWithRotations(Box, Wpt.GetRotations(params), false, nil)
		Mul := NewMultiplier(1)
		//we need to explicitly prepack cleartext input
		PrepackBlocks(Xenc, Wpt.InnerCols, Box)
		start := time.Now()
		resEnc := Mul.Multiply(Xenc, Wpt, true, Box)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.001, Box)
	})

	t.Run("Test/P2C_Multithread", func(t *testing.T) {
		S := NewSplitter(pU.NumRows(X), pU.NumCols(X), []int{pU.NumRows(W)}, []int{pU.NumCols(W)}, params)
		splits := S.FindSplits()
		splits.Print()
		Xenc, _ := NewPlainInput(X, splits.ExctractInfoAt(0)[2], splits.ExctractInfoAt(0)[3], params.MaxLevel(), params.DefaultScale(), Box)
		Wpt, _ := NewEncWeightDiag(W, splits.ExctractInfoAt(1)[2], splits.ExctractInfoAt(1)[3], Xenc.InnerRows, Xenc.InnerCols, params.MaxLevel(), Box)
		Box = BoxWithRotations(Box, Wpt.GetRotations(params), false, nil)
		Mul := NewMultiplier(runtime.NumCPU())
		start := time.Now()
		//we need to explicitly prepack cleartext input
		PrepackBlocks(Xenc, Wpt.InnerCols, Box)
		resEnc := Mul.Multiply(Xenc, Wpt, true, Box)
		fmt.Println("Done: ", time.Since(start))
		var res mat.Dense
		res.Mul(X, W)
		resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
		PrintDebugBlocks(resEnc, resPt, 0.001, Box)
	})

}

func TestAdder_AddBias(t *testing.T) {
	X := pU.RandMatrix(512, 512)
	B := pU.RandMatrix(512, 512)
	params, _ := ckks.NewParametersFromLiteral(bootstrapping.N16QP1546H192H32.SchemeParams)
	maxLevel := 9
	Box := NewBox(params)

	t.Run("Test/AddC2P", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, 8, 8, maxLevel, params.DefaultScale(), Box)
		Bpt, _ := NewPlainInput(B, 8, 8, maxLevel, params.DefaultScale(), Box)
		Ad := NewAdder(1)
		start := time.Now()
		Ad.AddBias(Xenc, Bpt, Box)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Add(X, B)
		resPt, _ := pU.PartitionMatrix(&res, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
	t.Run("Test/AddC2C", func(t *testing.T) {
		Xenc, _ := NewEncInput(X, 8, 8, maxLevel, params.DefaultScale(), Box)
		Bpt, _ := NewEncInput(B, 8, 8, maxLevel, params.DefaultScale(), Box)
		Ad := NewAdder(1)
		start := time.Now()
		Ad.AddBias(Xenc, Bpt, Box)
		fmt.Println("Done: ", time.Since(start))

		var res mat.Dense
		res.Add(X, B)
		resPt, _ := pU.PartitionMatrix(&res, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
	t.Run("Test/Add/Multitrehad", func(t *testing.T) {
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")
		Xenc, _ := NewEncInput(X, 8, 8, maxLevel, params.DefaultScale(), Box)
		Bpt, _ := NewPlainInput(B, 8, 8, maxLevel, params.DefaultScale(), Box)
		Ad := NewAdder(runtime.NumCPU())
		start := time.Now()
		Ad.AddBias(Xenc, Bpt, Box)
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
		Act, _ := NewActivator(1, 1)
		Act.AddActivation(*activation, 0, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box)
		start := time.Now()
		Act.ActivateBlocks(Xenc, 0, Box)
		fmt.Println("Done: ", time.Since(start))

		activation.ActivatePlain(Xscaled)

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})

	t.Run("Test/Activate/MultiThread", func(t *testing.T) {
		activation := utils.InitReLU(3)
		fmt.Println("Running on:", runtime.NumCPU(), "logical CPUs")
		// we need to rescale the input before the activation
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(1, runtime.NumCPU())
		Act.AddActivation(*activation, 0, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box)

		start := time.Now()
		Act.ActivateBlocks(Xenc, 0, Box)
		fmt.Println("Done: ", time.Since(start))

		activation.ActivatePlain(Xscaled) //this automatically rescales the input before activating

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})

	t.Run("Test/Activate/Chebybase", func(t *testing.T) {
		activation := utils.InitActivationCheby("silu", -5, 5, 10)
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(1, 1)
		Act.AddActivation(*activation, 0, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box)

		start := time.Now()
		Act.ActivateBlocks(Xenc, 0, Box)
		fmt.Println("Done: ", time.Since(start))

		activation.ActivatePlain(Xscaled) //this automatically rescales the input before activating

		resPt, _ := pU.PartitionMatrix(Xscaled, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})

	t.Run("Test/Activate/MultiThread/Chebybase", func(t *testing.T) {
		activation := utils.InitActivationCheby("silu", -5, 5, 10)
		Xscaled, _ := activation.Rescale(X, X)
		Xenc, _ := NewEncInput(Xscaled, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Act, _ := NewActivator(1, runtime.NumCPU())
		Act.AddActivation(*activation, 0, params.MaxLevel(), params.DefaultScale(), Xenc.InnerRows, Xenc.InnerCols, Box)

		start := time.Now()
		Act.ActivateBlocks(Xenc, 0, Box)
		fmt.Println("Done: ", time.Since(start))

		activation.ActivatePlain(Xscaled) //this automatically rescales the input before activating

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
	Box = BoxWithRotations(Box, nil, true, &btpParams)

	t.Run("Test/Bootstrap", func(t *testing.T) {

		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Btp := NewBootstrapper(1)
		start := time.Now()
		Btp.Bootstrap(Xenc, Box)
		fmt.Println("Done: ", time.Since(start))

		Xc := pU.NewDense(pU.MatToArray(X))

		resPt, _ := pU.PartitionMatrix(Xc, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
	t.Run("Test/Bootstrap/MultiThread", func(t *testing.T) {

		Xenc, _ := NewEncInput(X, 4, 4, params.MaxLevel(), params.DefaultScale(), Box)
		Btp := NewBootstrapper(runtime.NumCPU())
		start := time.Now()
		Btp.Bootstrap(Xenc, Box)
		fmt.Println("Done: ", time.Since(start))

		Xc := pU.NewDense(pU.MatToArray(X))

		resPt, _ := pU.PartitionMatrix(Xc, Xenc.RowP, Xenc.ColP)
		PrintDebugBlocks(Xenc, resPt, 0.01, Box)
	})
}

/*
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
	Mul := NewMultiplier( runtime.NumCPU())
	start = time.Now()
	resEnc := Mul.Multiply(Xenc, Wpt, true)
	fmt.Println("Done: ", time.Since(start))

	var res mat.Dense
	res.Mul(X, W)
	resPt, _ := pU.PartitionMatrix(&res, resEnc.RowP, resEnc.ColP)
	PrintDebugBlocks(resEnc, resPt, 0.01, Box)
}

//helpers

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
