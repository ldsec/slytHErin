package multidim

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"math"
	"testing"
	"time"
)

func Test_Multiplication(t *testing.T) {
	/*
		=== RUN   Test_Multiplication (256x256 x 256x256)
			--- PASS: Test_Multiplication (447.48s)
			=== RUN   Test_Multiplication/Test/C2P
			Parallel Batches:  8

			Possible split 1
			Splits for  Input
			InR: 32 InC: 128 RP: 8 CP: 2
			Splits for  Weight 1
			InR: 128 InC: 64 RP: 2 CP: 4

			Possible split 2
			Splits for  Input
			InR: 64 InC: 64 RP: 4 CP: 4
			Splits for  Weight 1
			InR: 64 InC: 64 RP: 4 CP: 4
			MultiDim:  6.0547084s
			Blocks:  46.0755868s
			MultiDim:  6.4206201s
			Blocks:  27.1108793s
			    --- PASS: Test_Multiplication/Test/C2P (113.06s)
			=== RUN   Test_Multiplication/Test/C2C
			Parallel Batches:  8

			Possible split 1
			Splits for  Input
			InR: 32 InC: 128 RP: 8 CP: 2
			Splits for  Weight 1
			InR: 128 InC: 64 RP: 2 CP: 4

			Possible split 2
			Splits for  Input
			InR: 64 InC: 64 RP: 4 CP: 4
			Splits for  Weight 1
			InR: 64 InC: 64 RP: 4 CP: 4
			MultiDim:  1m27.2544526s
			Blocks:  1m2.4505149s
			MultiDim:  1m37.3221046s
			Blocks:  38.0225027s
			--- PASS: Test_Multiplication/Test/C2C (334.41s)
			PASS

			Multidim seems to perform better when model is in clear
			Blocks seems to perform better when model is encrypted
			--> not true if we set rows = 36 tho

	*/
	params2, _ := ckks2.NewParametersFromLiteral(ckks2.PN14QP438)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	A := plainUtils.RandMatrix(256, 784)
	B := plainUtils.RandMatrix(784, 100)
	dim := utils2.MinInt(int(math.Ceil(float64(params2.N())/(2.0*float64(plainUtils.NumRows(A))))), plainUtils.NumRows(A))
	fmt.Println("Dim: ", dim)
	t.Run("Test/C2P", func(t *testing.T) {

		Apacked := PackMatrixParallel(A, dim, params2.LogSlots())
		ApackedT := new(PackedMatrix)
		ApackedT.Transpose(Apacked)
		fmt.Println("Parallel Batches: ", ApackedT.n)
		Bpacked := PackMatrixParallelReplicated(B, dim, Apacked.n)
		BpackedT := new(PackedMatrix)
		BpackedT.Transpose(Bpacked)

		encoder2 := ckks2.NewEncoder(params2)

		// Keys
		kgen := ckks2.NewKeyGenerator(params2)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		//decryptor := ckks2.NewDecryptor(params2, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params2.Scale(),
			TargetScale: params2.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params2, mmLiteral, encoder2)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 2,
			InputScale:  params2.Scale(),
			TargetScale: params2.Scale(),
		}

		rotations := mm_1.Rotations(params2)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params2, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params2, dim, utils2.MaxInt(ApackedT.rows, BpackedT.rows), utils2.MaxInt(ApackedT.cols, BpackedT.cols), eval)
		ppm.AddMatrixOperation(mm_1)

		batchEncryptor := NewBatchEncryptor(params2, sk)
		ctAp := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params2.Scale(), ApackedT)
		ptBp := batchEncryptor.EncodeForLeftMul(lvl_W0, BpackedT)

		Box := cipherUtils.NewBox(params)
		allSplits := cipherUtils.FindSplits(plainUtils.NumRows(A), plainUtils.NumCols(A), []int{plainUtils.NumRows(B)}, []int{plainUtils.NumCols(B)}, params, true)
		if len(allSplits) == 0 {
			panic(errors.New("No splits"))
		}
		cipherUtils.PrintAllSplits(allSplits)
		for _, splits := range allSplits {
			info := cipherUtils.ExctractInfo(splits)
			Box = cipherUtils.BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, info.InputRows, info.InputCols, 1, info.RowsOfWeights, info.ColsOfWeights)
			ctAb, _ := cipherUtils.NewEncInput(A, info.InputRowP, info.InputColP, params.MaxLevel(), Box)
			ptBb, _ := cipherUtils.NewPlainWeightDiag(B, splits[1].RowP, splits[1].ColP, info.InputRows, params.MaxLevel(), Box)
			mul := cipherUtils.NewMultiplier(Box, 1)

			startMd := time.Now()
			resp := AllocateCiphertextBatchMatrix(ptBp.Rows(), ctAp.Cols(), dim, ctAp.Level(), params2)
			ppm.MulPlainLeft([]*PlaintextBatchMatrix{ptBp}, ctAp, dim, []*CiphertextBatchMatrix{resp})
			doneMd := time.Since(startMd)

			startB := time.Now()
			mul.Multiply(ctAb, ptBb)
			doneB := time.Since(startB)

			fmt.Println("MultiDim: ", doneMd)
			fmt.Println("Blocks: ", doneB)
		}
	})

	t.Run("Test/C2C", func(t *testing.T) {

		Apacked := PackMatrixParallel(A, dim, params2.LogSlots())
		fmt.Println("Parallel Batches: ", Apacked.n)
		Bpacked := PackMatrixParallelReplicated(B, dim, Apacked.n)

		encoder2 := ckks2.NewEncoder(params2)

		// Keys
		kgen := ckks2.NewKeyGenerator(params2)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		//decryptor := ckks2.NewDecryptor(params2, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params2.Scale(),
			TargetScale: params2.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params2, mmLiteral, encoder2)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 2,
			InputScale:  params2.Scale(),
			TargetScale: params2.Scale(),
		}

		rotations := mm_1.Rotations(params2)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params2, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params2, dim, utils2.MaxInt(Apacked.rows, Bpacked.rows), utils2.MaxInt(Apacked.cols, Bpacked.cols), eval)
		ppm.AddMatrixOperation(mm_1)

		batchEncryptor := NewBatchEncryptor(params2, sk)
		ctAp := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params2.Scale(), Apacked)
		ctBp := batchEncryptor.EncodeAndEncrypt(lvl_W0, params2.Scale(), Bpacked)

		Box := cipherUtils.NewBox(params)
		allSplits := cipherUtils.FindSplits(plainUtils.NumRows(A), plainUtils.NumCols(A), []int{plainUtils.NumRows(B)}, []int{plainUtils.NumCols(B)}, params, true)
		if len(allSplits) == 0 {
			panic(errors.New("No splits"))
		}
		cipherUtils.PrintAllSplits(allSplits)
		for _, splits := range allSplits {
			info := cipherUtils.ExctractInfo(splits)
			Box = cipherUtils.BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, info.InputRows, info.InputCols, 1, info.RowsOfWeights, info.ColsOfWeights)
			ctAb, _ := cipherUtils.NewEncInput(A, info.InputRowP, info.InputColP, params.MaxLevel(), Box)
			ctBb, _ := cipherUtils.NewEncWeightDiag(B, splits[1].RowP, splits[1].ColP, info.InputRows, params.MaxLevel(), Box)
			mul := cipherUtils.NewMultiplier(Box, 1)

			startMd := time.Now()
			resp := AllocateCiphertextBatchMatrix(ctAp.Rows(), ctBp.Cols(), dim, ctAp.Level(), params2)
			ppm.MulSquareMatricesPacked(ctAp, ctBp, dim, resp)
			doneMd := time.Since(startMd)

			startB := time.Now()
			mul.Multiply(ctAb, ctBb)
			doneB := time.Since(startB)

			fmt.Println("MultiDim: ", doneMd)
			fmt.Println("Blocks: ", doneB)
		}
	})
}
