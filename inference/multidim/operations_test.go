package multidim

import (
	"fmt"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
	"time"
)

func TestPackingSingle(t *testing.T) {
	rows := 784
	cols := 784
	dim := 28
	x := make([]float64, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			x[row*cols+col] = float64(row)
		}
	}
	x_mat := mat.NewDense(rows, cols, x)
	pm := PackMatrixSingle(x_mat, dim)
	m := UnpackMatrixSingle(pm, dim, rows, cols)
	new_mat := mat.NewDense(rows, cols, m)
	pm.Print(0)
	MatrixPrint(new_mat)

	// Schemes parameters are created from scratch
	params, err := ckks2.NewParametersFromLiteral(ckks2.ParametersLiteral{
		LogN:     14,
		LogQ:     []int{60, 35, 35, 35},
		LogP:     []int{61, 61},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 13,
		Scale:    float64(1 << 35),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(params.LogQP())

	encoder := ckks2.NewEncoder(params)

	// Keys
	kgen := ckks2.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPair()

	// Relinearization key
	rlk := kgen.GenRelinearizationKey(sk, 2)

	// Decryptor
	decryptor := ckks2.NewDecryptor(params, sk)

	// var mat_size int = 128

	// Size of the matrices (dxd)

	rows0 := 28
	cols0 := 28
	rows1 := 28
	cols1 := 10

	lvl_W0 := params.MaxLevel()
	mmLiteral := MatrixMultiplicationLiteral{
		Dimension:   dim,
		LevelStart:  lvl_W0,
		InputScale:  params.Scale(),
		TargetScale: params.Scale(),
	}
	mmW0 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
	transposeLT := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
	// Rotation-keys generation
	rotations := mmW0.Rotations(params)
	rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT.PtDiagMatrix)...)
	rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
	ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(rows0, rows1), utils2.MaxInt(cols0, cols1), eval)
	ppm.AddMatrixOperation(mmW0)
	ppm.AddMatrixOperation(transposeLT)
	batchEncryptor := NewBatchEncryptor(params, sk)

	ct := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), pm)
	mm := UnpackCipherSingle(ct, dim, rows, cols, encoder, decryptor, params)
	unpacked_mat := mat.NewDense(rows, cols, mm)
	MatrixPrint(unpacked_mat)
	pt := batchEncryptor.EncodeSingle(params.MaxLevel(), params.Scale(), pm)
	mmm := UnpackPlainSingle(pt, dim, rows, cols, encoder, params)
	unpacked_mmat := mat.NewDense(rows, cols, mmm)
	MatrixPrint(unpacked_mmat)

}

func TestPackingParallel(t *testing.T) {
	rows := 784
	cols := 100
	dim := 28
	x := make([]float64, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			x[row*cols+col] = float64(row) + float64(col)/100
		}
	}
	params, err := ckks2.NewParametersFromLiteral(ckks2.ParametersLiteral{
		LogN:     14,
		LogQ:     []int{60, 35, 35, 35},
		LogP:     []int{61, 61},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 13,
		Scale:    float64(1 << 35),
	})
	if err != nil {
		panic(err)
	}

	x_mat := mat.NewDense(rows, cols, x)
	pm := PackMatrixParallel(x_mat, dim, params.LogSlots())
	n := (1 << params.LogSlots()) / (dim * dim)
	m := UnpackMatrixParallel(pm, dim, rows, cols)
	new_mat := mat.NewDense(rows, cols, m)
	pm.Print(0)
	pm.Print(1)
	pm.Print(2)
	pm.Print(3)

	MatrixPrint(new_mat)

	fmt.Println(params.LogQP())

	encoder := ckks2.NewEncoder(params)

	// Keys
	kgen := ckks2.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPair()

	// Relinearization key
	//rlk := kgen.GenRelinearizationKey(sk, 2)

	// Decryptor
	decryptor := ckks2.NewDecryptor(params, sk)

	// var mat_size int = 128

	// Size of the matrices (dxd)

	//rows0 := 28
	//cols0 := 28
	//rows1 := 28
	//cols1 := 28

	lvl_W0 := params.MaxLevel()
	mmLiteral := MatrixMultiplicationLiteral{
		Dimension:   dim,
		LevelStart:  lvl_W0,
		InputScale:  params.Scale(),
		TargetScale: params.Scale(),
	}
	mmW0 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
	transposeLT := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
	// Rotation-keys generation
	rotations := mmW0.Rotations(params)
	rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT.PtDiagMatrix)...)
	//rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	//eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
	//ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(rows0, rows1), utils2.MaxInt(cols0, cols1), eval)
	//ppm.AddMatrixOperation(mmW0)
	//ppm.AddMatrixOperation(transposeLT)
	batchEncryptor := NewBatchEncryptor(params, sk)

	ct := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), pm)
	mm := UnpackCipherParallel(ct, dim, rows, cols, encoder, decryptor, params, n)
	unpacked_mat := mat.NewDense(rows, cols, mm)
	MatrixPrint(unpacked_mat)
	pt := batchEncryptor.EncodeParallel(params.MaxLevel(), params.Scale(), pm)
	mmm := UnpackPlainParallel(pt, dim, rows, cols, encoder, params, n)
	unpacked_mmat := mat.NewDense(rows, cols, mmm)
	MatrixPrint(unpacked_mmat)

}

func TestPackedMatrices(t *testing.T) {

	// Schemes parameters are created from scratch
	params, err := ckks2.NewParametersFromLiteral(ckks2.ParametersLiteral{
		LogN:     14,
		LogQ:     []int{60, 35, 35, 35},
		LogP:     []int{61, 61},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 13,
		Scale:    float64(1 << 35),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(params.LogQP())

	encoder := ckks2.NewEncoder(params)

	// Keys
	kgen := ckks2.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPair()

	// Relinearization key
	rlk := kgen.GenRelinearizationKey(sk, 2)

	// Encryptor
	encryptor := ckks2.NewEncryptor(params, pk)

	// Decryptor
	decryptor := ckks2.NewDecryptor(params, sk)

	// var mat_size int = 128

	// Size of the matrices (dxd)
	var dim int = 4
	var parallelBatches int = 4

	rows0 := 8 * parallelBatches
	cols0 := 8
	rows1 := 8
	cols1 := 6

	lvl_W0 := params.MaxLevel()

	mmLiteral := MatrixMultiplicationLiteral{
		Dimension:   dim,
		LevelStart:  lvl_W0,
		InputScale:  params.Scale(),
		TargetScale: params.Scale(),
	}

	mmW0 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
	transposeLT := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)

	// Rotation-keys generation
	rotations := mmW0.Rotations(params)
	rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT.PtDiagMatrix)...)
	rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)

	eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})

	ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(rows0, rows1), utils2.MaxInt(cols0, cols1), eval)

	ppm.AddMatrixOperation(mmW0)
	ppm.AddMatrixOperation(transposeLT)

	batchEncryptor := NewBatchEncryptor(params, sk)

	t.Run("Packed/Multiply/Encrypted", func(t *testing.T) {

		W0 := GenRandomRealPackedMatrices(dim, rows0, cols0, parallelBatches)
		W1 := GenRandomRealPackedMatrices(dim, rows1, cols1, parallelBatches)

		matCtW0 := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), W0)
		matCtW1 := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), W1)

		res := AllocateCiphertextBatchMatrix(rows0, cols1, dim, matCtW0.Level()-2, params)

		resPlain := new(PackedMatrix)
		resPlain.Mul(W0, W1)

		start := time.Now()
		ppm.MulSquareMatricesPacked(matCtW0, matCtW1, dim, res)
		fmt.Printf("Since: %s\n", time.Since(start))

		for i := range res.M {
			VerifyTestVectors(params, encoder, decryptor, resPlain.M[i], res.M[i], t)
		}
	})

	// t.Run("Packed/Multiply/Strassen", func(t *testing.T) {

	// 	W0 := GenRandomRealPackedMatrices(dim, rows0, cols0, parallelBatches)
	// 	W1 := GenRandomRealPackedMatrices(dim, rows1, cols1, parallelBatches)

	// 	matCtW0 := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), W0)
	// 	matCtW1 := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), W1)

	// 	res := AllocateCiphertextBatchMatrix(rows0, cols1, dim, matCtW0.Level()-2, params)

	// 	resPlain := new(PackedMatrix)
	// 	resPlain.Mul(W0, W1)

	// 	start := time.Now()
	// 	ppm.Strassen(matCtW0, matCtW1, dim, res)
	// 	fmt.Printf("Since: %s\n", time.Since(start))

	// 	for i := range res.M {
	// 		VerifyTestVectors(params, encoder, decryptor, resPlain.M[i], res.M[i], t)
	// 	}
	// })

	t.Run("Packed/Multiply/Plain", func(t *testing.T) {
		W0 := make([]*PackedMatrix, 1)
		for i := range W0 {
			W0[i] = GenRandomRealPackedMatrices(dim, rows0, cols0, parallelBatches)
		}
		W1 := GenRandomRealPackedMatrices(dim, rows1, cols1, parallelBatches)

		matPtW0 := make([]*PlaintextBatchMatrix, len(W0))
		for i := range matPtW0 {
			matPtW0[i] = batchEncryptor.EncodeForLeftMul(params.MaxLevel(), W0[i])
		}
		matCtW1 := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), W1)

		res := make([]*CiphertextBatchMatrix, len(W0))
		for i := range res {
			res[i] = AllocateCiphertextBatchMatrix(rows0, cols1, dim, matCtW1.Level()-1, params)
		}

		resPlain := make([]*PackedMatrix, len(W0))
		for i := range resPlain {
			resPlain[i] = new(PackedMatrix)
			resPlain[i].Mul(W0[i], W1)
		}

		start := time.Now()
		ppm.MulPlainLeft(matPtW0, matCtW1, dim, res)
		fmt.Printf("Since: %s\n", time.Since(start))

		for i := range resPlain {
			for j := range resPlain[i].M {
				VerifyTestVectors(params, encoder, decryptor, resPlain[i].M[j], res[i].M[j], t)
			}
		}
	})

	t.Run("Packed/Transpose/", func(t *testing.T) {

		W0 := GenRandomRealPackedMatrices(dim, rows0, cols0, parallelBatches)

		ctW0 := make([]*ckks2.Ciphertext, rows0*cols0)

		pt := ckks2.NewPlaintext(params, lvl_W0, params.Scale())
		values := make([]complex128, params.Slots())
		for i := 0; i < rows0*cols0; i++ {
			EncodeMatrices(values, pt, W0.M[i], encoder, params.LogSlots())
			ctW0[i] = encryptor.EncryptNew(pt)
		}

		matCtW0 := NewCiphertextBatchMatrix(rows0, cols0, dim, ctW0)

		/*
			for i := range matCtW0.M {
				decryptPrint(parallelBatches, dim, dim, matCtW0.M[i], decryptor, encoder, params.LogSlots())
			}
			fmt.Println()
		*/

		start := time.Now()
		ppm.Transpose(matCtW0, matCtW0)
		fmt.Printf("Since :%s\n", time.Since(start))

		resPlain := new(PackedMatrix)
		resPlain.Transpose(W0)

		fmt.Println(matCtW0)

		for i := range matCtW0.M {
			//decryptPrint(parallelBatches, dim, dim, matCtW0.M[i], decryptor, encoder, params.LogSlots())
			VerifyTestVectors(params, encoder, decryptor, resPlain.M[i], matCtW0.M[i], t)
		}
	})
	t.Run("test", func(t *testing.T) {
		rows := 16
		cols := 16
		dim := 4
		n := (1 << 6) / (dim * dim)
		x := make([]float64, rows*cols)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				x[row*cols+col] = float64(row) + float64(col)/100
			}
		}
		x_mat := mat.NewDense(rows, cols, x)
		pmx := PackMatrixParallel(x_mat, dim, 6)

		y := make([]float64, rows*cols)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				y[row*cols+col] = float64(row)
			}
		}
		y_mat := mat.NewDense(rows, cols, y)
		pmy := PackMatrixParallel(y_mat, dim, 6)
		Xpt := make([]*PlaintextBatchMatrix, 1)
		Xpt[0] = batchEncryptor.EncodeForLeftMul(params.MaxLevel(), pmx)
		Yct := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), pmy)

		Resct := make([]*CiphertextBatchMatrix, 1)
		Resct[0] = AllocateCiphertextBatchMatrix(rows, cols, dim, Yct.Level()-1, params)
		ppm.MulPlainLeft(Xpt, Yct, dim, Resct)

		res := UnpackCipherParallel(Resct[0], dim, rows, cols, encoder, decryptor, params, n)
		MatrixPrint(mat.NewDense(rows, cols, res))
		z_mat := mat.NewDense(rows, cols, nil)
		z_mat.Mul(x_mat, y_mat)
		MatrixPrint(z_mat)

	})
}

func decryptPrint(n, rows, cols int, ciphertext *ckks2.Ciphertext, decryptor ckks2.Decryptor, encoder ckks2.Encoder, logSlots int) {
	v := encoder.Decode(decryptor.DecryptNew(ciphertext), logSlots)[:rows*cols]

	for x := 0; x < n; x++ {
		idx := x * rows * cols
		fmt.Printf("[\n")
		for i := 0; i < rows; i++ {
			fmt.Printf("[ ")
			for j := 0; j < cols; j++ {
				fmt.Printf("%7.4f, ", real(v[idx+i*cols+j]))
			}
			fmt.Printf("],\n")
		}
		fmt.Printf("]\n")
	}
}
