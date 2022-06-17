package multidim

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
	"time"
)

func Test_MultiDimPacking_Operations(t *testing.T) {
	ckksParams := ckks2.ParametersLiteral{
		LogN:     14,
		LogQ:     []int{40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 218 for LogN 13
		LogP:     []int{33, 33, 33},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 13,
		Scale:    float64(1 << 30),
	}
	params, _ := ckks2.NewParametersFromLiteral(ckksParams)

	A := plainUtils.RandMatrix(64, 64)
	B := plainUtils.RandMatrix(64, 32)
	C := plainUtils.RandMatrix(32, 16)

	dim := 8

	t.Run("Test/Mult/Plain/Single", func(t *testing.T) {
		Apacked := PackMatrixSingle(A, dim)
		Bpacked := PackMatrixSingle(B, dim)
		Cpacked := PackMatrixSingle(C, dim)

		var tmp mat.Dense
		var res mat.Dense
		tmp.Mul(A, B)
		res.Mul(&tmp, C)

		Apacked.Mul(Apacked, Bpacked)
		Cpacked.Mul(Apacked, Cpacked)

		resFromPack := UnpackMatrixSingle(Cpacked, dim, plainUtils.NumRows(&res), plainUtils.NumCols(&res))
		resFromDense := plainUtils.Vectorize(plainUtils.MatToArray(&res), true)
		fmt.Println("Pack:", len(resFromPack), "dense", len(resFromDense))
		for i := range resFromDense {
			fmt.Println("Test:", resFromPack[i])
			fmt.Println("Want:", resFromDense[i])
			require.LessOrEqual(t, math.Abs(resFromPack[i]-resFromDense[i]), 1e-10)
		}
	})

	t.Run("Test/Mult/Enc/Single", func(t *testing.T) {
		Apacked := PackMatrixSingle(A, dim)
		Bpacked := PackMatrixSingle(B, dim)
		Cpacked := PackMatrixSingle(C, dim)

		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 3,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_2 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		transposeLT_2 := GenTransposeDiagMatrix(params.MaxLevel()-3, 1.0, 4.0, dim, params, encoder)

		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		rotations = append(rotations, mm_2.Rotations(params)...)
		rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(utils2.MaxInt(Apacked.rows, Bpacked.rows), Cpacked.rows), utils2.MaxInt(utils2.MaxInt(Apacked.cols, Bpacked.cols), Cpacked.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		ppm.AddMatrixOperation(transposeLT_1)
		ppm.AddMatrixOperation(mm_2)
		ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Apacked)
		ctB := batchEncryptor.EncodeAndEncrypt(lvl_W0, params.Scale(), Bpacked)

		ctC := batchEncryptor.EncodeAndEncrypt(lvl_W0-3, params.Scale(), Cpacked)

		Apacked.Mul(Apacked, Bpacked)
		Cpacked.Mul(Apacked, Cpacked)

		start := time.Now()
		ctTmp := AllocateCiphertextBatchMatrix(ctA.Rows(), ctB.Cols(), dim, ctA.Level()-2, params)
		ppm.MulSquareMatricesPacked(ctA, ctB, dim, ctTmp)
		ctRes := AllocateCiphertextBatchMatrix(ctTmp.Rows(), ctC.Cols(), dim, ctTmp.Level()-2, params)
		fmt.Println("level: ", ctTmp.Level())
		ppm.MulSquareMatricesPacked(ctTmp, ctC, dim, ctRes)
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		resPlain := UnpackMatrixSingle(Cpacked, dim, plainUtils.NumRows(A), plainUtils.NumCols(C))
		resCipher := UnpackCipherSingle(ctRes, dim, plainUtils.NumRows(A), plainUtils.NumCols(C), encoder, decryptor, params)
		for i := range resPlain {
			fmt.Println("Test:", resCipher[i])
			fmt.Println("Want:", resPlain[i])
			require.LessOrEqual(t, math.Abs(resPlain[i]-resCipher[i]), 1e-1)
		}
	})

	t.Run("Test/Mult/Plain/Parallel", func(t *testing.T) {
		Apacked := PackMatrixParallel(A, dim, params.LogSlots())
		Bpacked := PackMatrixParallelReplicated(B, dim, Apacked.n)
		Cpacked := PackMatrixParallelReplicated(C, dim, Apacked.n)

		var tmp mat.Dense
		var res mat.Dense
		tmp.Mul(A, B)
		res.Mul(&tmp, C)

		Apacked.Mul(Apacked, Bpacked)
		Cpacked.Mul(Apacked, Cpacked)

		resFromPack := UnpackMatrixParallel(Cpacked, dim, plainUtils.NumRows(&res), plainUtils.NumCols(&res))
		resFromDense := plainUtils.Vectorize(plainUtils.MatToArray(&res), true)
		fmt.Println("Pack:", len(resFromPack), "dense", len(resFromDense))
		for i := range resFromDense {
			fmt.Println("Test:", resFromPack[i])
			fmt.Println("Want:", resFromDense[i])
			require.LessOrEqual(t, math.Abs(resFromPack[i]-resFromDense[i]), 1e-10)
		}
	})

	t.Run("Test/Mult/Enc/Parallel", func(t *testing.T) {
		Apacked := PackMatrixParallel(A, dim, params.LogSlots())
		fmt.Println("Parallel Batches: ", Apacked.n)
		Bpacked := PackMatrixParallelReplicated(B, dim, Apacked.n)
		Cpacked := PackMatrixParallelReplicated(C, dim, Apacked.n)

		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 3,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_2 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		transposeLT_2 := GenTransposeDiagMatrix(params.MaxLevel()-3, 1.0, 4.0, dim, params, encoder)
		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		rotations = append(rotations, mm_2.Rotations(params)...)
		rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(utils2.MaxInt(Apacked.rows, Bpacked.rows), Cpacked.rows), utils2.MaxInt(utils2.MaxInt(Apacked.cols, Bpacked.cols), Cpacked.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		ppm.AddMatrixOperation(transposeLT_1)
		ppm.AddMatrixOperation(mm_2)
		ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Apacked)
		ctB := batchEncryptor.EncodeAndEncrypt(lvl_W0, params.Scale(), Bpacked)
		ctC := batchEncryptor.EncodeAndEncrypt(lvl_W0-3, params.Scale(), Cpacked)

		Apacked.Mul(Apacked, Bpacked)
		Cpacked.Mul(Apacked, Cpacked)

		start := time.Now()
		ctTmp := AllocateCiphertextBatchMatrix(ctA.Rows(), ctB.Cols(), dim, ctA.Level()-2, params)
		ppm.MulSquareMatricesPacked(ctA, ctB, dim, ctTmp)

		resPlain := UnpackMatrixParallel(Apacked, dim, plainUtils.NumRows(A), plainUtils.NumCols(B))
		resCipher := UnpackCipherParallel(ctTmp, dim, plainUtils.NumRows(A), plainUtils.NumCols(B), encoder, decryptor, params, Apacked.n)
		for i := range resPlain {
			fmt.Println("Test:", resCipher[i])
			fmt.Println("Want:", resPlain[i])
			require.LessOrEqual(t, math.Abs(resPlain[i]-resCipher[i]), 1e-1)
		}

		ctRes := AllocateCiphertextBatchMatrix(ctTmp.Rows(), ctC.Cols(), dim, 1, params)
		fmt.Println("level: ", ctTmp.Level())
		ppm.MulSquareMatricesPacked(ctTmp, ctC, dim, ctRes)
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		resPlain2 := UnpackMatrixParallel(Cpacked, dim, plainUtils.NumRows(A), plainUtils.NumCols(C))
		resCipher2 := UnpackCipherParallel(ctRes, dim, plainUtils.NumRows(A), plainUtils.NumCols(C), encoder, decryptor, params, Apacked.n)
		for i := range resPlain2 {
			fmt.Println("Test:", resCipher2[i])
			fmt.Println("Want:", resPlain2[i])
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 1e-1)
		}
	})
	t.Run("Test/Mult/Hybrid/MulPtLeft", func(t *testing.T) {
		//pt x ct
		D := plainUtils.RandMatrix(4, 4)
		E := plainUtils.RandMatrix(4, 4)
		Dpacked := PackMatrixSingle(D, dim)
		Epacked := PackMatrixParallelReplicated(E, dim, Dpacked.n)

		//E x D
		var res mat.Dense
		res.Mul(E, D)

		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(Dpacked.rows, Epacked.rows), utils2.MaxInt(Dpacked.cols, Epacked.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		//ppm.AddMatrixOperation(transposeLT_1)
		//ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctD := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Dpacked)
		ptE := batchEncryptor.EncodeForLeftMul(lvl_W0, Epacked)

		ctRes := AllocateCiphertextBatchMatrix(Epacked.Rows(), Dpacked.Cols(), dim, lvl_W0, params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ptE}, ctD, dim, []*CiphertextBatchMatrix{ctRes})

		resCipher2 := UnpackCipherParallel(ctRes, dim, plainUtils.NumRows(E), plainUtils.NumCols(D), encoder, decryptor, params, Dpacked.n)
		resPlain2 := plainUtils.RowFlatten(&res)
		for i := range resPlain2 {
			fmt.Println("Test:", resCipher2[i])
			fmt.Println("Want:", resPlain2[i])
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 1e-1)
		}
	})

	t.Run("Test/Mult/Hybrid/MulPtLeftWithTranspose", func(t *testing.T) {
		//pt x ct
		D := plainUtils.RandMatrix(64, 64) //->ct
		E := plainUtils.RandMatrix(64, 64) //->pt
		Dpacked := PackMatrixSingle(D, dim)
		DpackedT := new(PackedMatrix)
		DpackedT.Transpose(Dpacked)
		Epacked := PackMatrixParallelReplicated(E, dim, Dpacked.n)
		EpackedT := new(PackedMatrix)
		EpackedT.Transpose(Epacked)

		//D x E == (E.T x D.T).T
		var res mat.Dense
		res.Mul(D, E)

		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(Dpacked.rows, Epacked.rows), utils2.MaxInt(Dpacked.cols, Epacked.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		//ppm.AddMatrixOperation(transposeLT_1)
		//ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctD := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), DpackedT)
		ptE := batchEncryptor.EncodeForLeftMul(lvl_W0, EpackedT)

		ctRes := AllocateCiphertextBatchMatrix(Epacked.Rows(), Dpacked.Cols(), dim, lvl_W0, params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ptE}, ctD, dim, []*CiphertextBatchMatrix{ctRes})

		resCipher := UnpackCipherParallel(ctRes, dim, plainUtils.NumCols(E), plainUtils.NumRows(D), encoder, decryptor, params, Dpacked.n)
		resCipher2 := plainUtils.RowFlatten(plainUtils.TransposeDense(mat.NewDense(plainUtils.NumCols(E), plainUtils.NumRows(D), resCipher)))
		//resPlain2 := plainUtils.RowFlatten(plainUtils.TransposeDense(&res))
		resPlain2 := plainUtils.RowFlatten(&res)
		for i := range resPlain2 {
			fmt.Println("Test:", resCipher2[i])
			fmt.Println("Want:", resPlain2[i])
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 1e-1)
		}
	})

	t.Run("Test/Mult/Hybrid/Single", func(t *testing.T) {
		//pt x ct

		// A x B x C = (C.T X B.T X A.T).T = ((A x B X C).T).T
		//if Parallel bad things happen, probably because 64x64 is too small for parallel batches
		Apacked := PackMatrixSingle(A, dim)
		ApackedT := new(PackedMatrix)
		ApackedT.Transpose(Apacked)
		fmt.Println("Parallel Batches: ", ApackedT.n)
		Bpacked := PackMatrixParallelReplicated(B, dim, Apacked.n)
		BpackedT := new(PackedMatrix)
		BpackedT.Transpose(Bpacked)
		Cpacked := PackMatrixParallelReplicated(C, dim, Apacked.n)
		CpackedT := new(PackedMatrix)
		CpackedT.Transpose(Cpacked)
		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 2,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_2 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_2 := GenTransposeDiagMatrix(params.MaxLevel()-2, 1.0, 4.0, dim, params, encoder)
		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		rotations = append(rotations, mm_2.Rotations(params)...)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(utils2.MaxInt(ApackedT.rows, BpackedT.rows), CpackedT.rows), utils2.MaxInt(utils2.MaxInt(ApackedT.cols, BpackedT.cols), CpackedT.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		ppm.AddMatrixOperation(mm_2)
		//ppm.AddMatrixOperation(transposeLT_1)
		//ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), ApackedT)
		ctB := batchEncryptor.EncodeForLeftMul(lvl_W0, BpackedT)
		ctC := batchEncryptor.EncodeForLeftMul(lvl_W0-2, CpackedT)

		start := time.Now()
		ctTmp := AllocateCiphertextBatchMatrix(ctB.Rows(), ctA.Cols(), dim, ctA.Level(), params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ctB}, ctA, dim, []*CiphertextBatchMatrix{ctTmp})
		ctRes := AllocateCiphertextBatchMatrix(ctC.Rows(), ctTmp.Cols(), dim, ctA.Level(), params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ctC}, ctTmp, dim, []*CiphertextBatchMatrix{ctRes})
		fmt.Println("level: ", ctRes.Level())
		require.Equal(t, lvl_W0-ctRes.Level(), 4)
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		var tmp mat.Dense
		var res mat.Dense
		tmp.Mul(A, B)
		res.Mul(&tmp, C)

		resPlain2 := plainUtils.RowFlatten(&res)
		//transpose this
		resCipher := UnpackCipherParallel(ctRes, dim, plainUtils.NumCols(C), plainUtils.NumRows(A), encoder, decryptor, params, Apacked.n)
		resCipher2 := plainUtils.RowFlatten(plainUtils.TransposeDense(mat.NewDense(plainUtils.NumCols(C), plainUtils.NumRows(A), resCipher)))
		for i := range resPlain2 {
			fmt.Println("Test ", i, " :", resCipher2[i])
			fmt.Println("Want ", i, " :", resPlain2[i])
			fmt.Println()
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 1e-1)
		}
	})

	t.Run("Test/Mult/Hybrid/Parallel", func(t *testing.T) {
		//pt x ct

		// A x B x C = (C.T X B.T X A.T).T = ((A x B X C).T).T
		//dim = int(math.Ceil(float64(params.MaxSlots()) / float64(plainUtils.NumRows(A))))
		Apacked := PackMatrixParallelReplicated(A, dim, params.LogSlots())
		ApackedT := new(PackedMatrix)
		ApackedT.Transpose(Apacked)
		fmt.Println("Parallel Batches: ", ApackedT.n)
		Bpacked := PackMatrixParallelReplicated(B, dim, params.LogSlots())
		BpackedT := new(PackedMatrix)
		BpackedT.Transpose(Bpacked)
		Cpacked := PackMatrixParallelReplicated(C, dim, params.LogSlots())
		CpackedT := new(PackedMatrix)
		CpackedT.Transpose(Cpacked)
		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 2,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_2 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_2 := GenTransposeDiagMatrix(params.MaxLevel()-2, 1.0, 4.0, dim, params, encoder)
		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		rotations = append(rotations, mm_2.Rotations(params)...)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		//rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(utils2.MaxInt(ApackedT.rows, BpackedT.rows), CpackedT.rows), utils2.MaxInt(utils2.MaxInt(ApackedT.cols, BpackedT.cols), CpackedT.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		ppm.AddMatrixOperation(mm_2)
		//ppm.AddMatrixOperation(transposeLT_1)
		//ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), ApackedT)
		ctB := batchEncryptor.EncodeForLeftMul(lvl_W0, BpackedT)
		ctC := batchEncryptor.EncodeForLeftMul(lvl_W0-2, CpackedT)

		start := time.Now()
		ctTmp := AllocateCiphertextBatchMatrix(ctB.Rows(), ctA.Cols(), dim, ctA.Level(), params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ctB}, ctA, dim, []*CiphertextBatchMatrix{ctTmp})
		ctRes := AllocateCiphertextBatchMatrix(ctC.Rows(), ctTmp.Cols(), dim, ctA.Level(), params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ctC}, ctTmp, dim, []*CiphertextBatchMatrix{ctRes})
		fmt.Println("level: ", ctRes.Level())
		require.Equal(t, lvl_W0-ctRes.Level(), 4)
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		var tmp mat.Dense
		var res mat.Dense
		tmp.Mul(A, B)
		res.Mul(&tmp, C)

		resPlain2 := plainUtils.RowFlatten(&res)
		fmt.Println("Plain")
		for i, r := range resPlain2 {
			fmt.Println(i, " --> ", r)
		}
		//transpose this
		resCipher := UnpackCipherParallel(ctRes, dim, plainUtils.NumRows(A), plainUtils.NumCols(C), encoder, decryptor, params, Apacked.n)
		fmt.Println("Cipher")
		for i, r := range resCipher {
			fmt.Println(i, " --> ", r)
		}
		resCipher2 := plainUtils.RowFlatten(plainUtils.TransposeDense(mat.NewDense(plainUtils.NumCols(C), plainUtils.NumRows(A), resCipher)))

		for i := range resCipher2 {
			fmt.Println("Test ", i, " :", resCipher2[i])
			fmt.Println("Want ", i, " :", resPlain2[i])
			fmt.Println()
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 0.5)
		}
	})

	t.Run("Test/Activation/Poly", func(t *testing.T) {

		// A x B x C = (C.T X B.T X A.T).T = ((A x B X C).T).T
		//if Parallel bad things happen, probably because 64x64 is too small for parallel batches
		Apacked := PackMatrixParallel(A, dim, params.LogSlots())

		activation := utils.InitReLU(3)
		f := func(v float64) float64 {
			return v / float64(activation.Interval)
		}
		Apacked.Apply(Apacked, f)
		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		// Rotation-keys generation

		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: nil})
		ppm := NewPackedMatrixMultiplier(params, dim, Apacked.rows, Apacked.cols, eval)
		//ppm.AddMatrixOperation(transposeLT_1)
		//ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Apacked)
		start := time.Now()
		ctRes := ppm.EvalPoly(ctA, ckks2.NewPoly(activation.Poly.Coeffs))
		fmt.Println("level: ", ctRes.Level())
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		utils.ActivatePlain(A, activation)
		//transpose this
		resCipher2 := UnpackCipherParallel(ctRes, dim, plainUtils.NumRows(A), plainUtils.NumCols(A), encoder, decryptor, params, Apacked.n)
		resPlain2 := plainUtils.RowFlatten(A)
		for i := range resPlain2 {
			fmt.Println("Test ", i, " :", resCipher2[i])
			fmt.Println("Want ", i, " :", resPlain2[i])
			fmt.Println()
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 1e-1)
		}
	})

	t.Run("Test/Mult/WBA/", func(t *testing.T) {
		//ct x Weight + Bias -> Activation

		// A x B x C = (C.T X B.T X A.T).T = ((A x B X C).T).T
		//if Parallel bad things happen, probably because 64x64 is too small for parallel batches
		Apacked := PackMatrixSingle(A, dim)
		//Apacked := PackMatrixParallel(A, dim, params.LogSlots())
		ApackedT := new(PackedMatrix)
		ApackedT.Transpose(Apacked)
		fmt.Println("Parallel Batches: ", ApackedT.n)
		W := plainUtils.RandMatrix(plainUtils.NumRows(A), 32)
		Bpacked := PackMatrixParallelReplicated(W, dim, Apacked.n)
		BpackedT := new(PackedMatrix)
		BpackedT.Transpose(Bpacked)
		Bias := plainUtils.RandMatrix(plainUtils.NumRows(A), plainUtils.NumCols(W))
		Cpacked := PackMatrixSingle(Bias, dim)
		CpackedT := new(PackedMatrix)
		CpackedT.Transpose(Cpacked)

		activation := utils.InitReLU(3)
		f := func(v float64) float64 {
			return v / float64(activation.Interval)
		}
		BpackedT.Apply(BpackedT, f)
		CpackedT.Apply(CpackedT, f)

		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		//transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(utils2.MaxInt(ApackedT.rows, BpackedT.rows), CpackedT.rows), utils2.MaxInt(utils2.MaxInt(ApackedT.cols, BpackedT.cols), CpackedT.cols), eval)
		ppm.AddMatrixOperation(mm_1)

		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), ApackedT)
		ctB := batchEncryptor.EncodeForLeftMul(lvl_W0, BpackedT)
		ctC := batchEncryptor.EncodeParallel(lvl_W0-2, params.Scale(), CpackedT)

		start := time.Now()
		ctTmp := AllocateCiphertextBatchMatrix(ctB.Rows(), ctA.Cols(), dim, ctA.Level(), params)
		ppm.MulPlainLeft([]*PlaintextBatchMatrix{ctB}, ctA, dim, []*CiphertextBatchMatrix{ctTmp})
		ctTmp2 := AllocateCiphertextBatchMatrix(ctC.Rows(), ctC.Cols(), dim, ctA.Level(), params)
		ppm.AddPlain(ctTmp, ctC, ctTmp2)
		ctRes := ppm.EvalPoly(ctTmp2, ckks2.NewPoly(activation.Poly.Coeffs))
		fmt.Println("level: ", ctRes.Level())
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		var tmp mat.Dense
		var res mat.Dense
		tmp.Mul(A, W)
		res.Add(&tmp, Bias)
		utils.ActivatePlain(&res, activation)

		resPlain2 := plainUtils.RowFlatten(&res)
		//transpose this
		resCipher := UnpackCipherParallel(ctRes, dim, plainUtils.NumCols(W), plainUtils.NumRows(A), encoder, decryptor, params, Apacked.n)
		resCipher2 := plainUtils.RowFlatten(plainUtils.TransposeDense(mat.NewDense(plainUtils.NumCols(W), plainUtils.NumRows(A), resCipher)))
		for i := range resPlain2 {
			fmt.Println("Test ", i, " :", resCipher2[i])
			fmt.Println("Want ", i, " :", resPlain2[i])
			fmt.Println()
			require.LessOrEqual(t, math.Abs(resPlain2[i]-resCipher2[i]), 1e-1)
		}
	})
}
