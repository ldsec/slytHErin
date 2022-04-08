package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"sync"
	"testing"
	"time"
)

/********************************************
BLOCK MATRICES OPS
|
|
v
*********************************************/
func TestDecInput(t *testing.T) {
	LDim := []int{64, 10}

	r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = r.NormFloat64()
		}
	}
	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 1, 1)

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
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

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})
	Box := CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}

	ctA, err := NewEncInput(L, 1, 1, Box)
	CompareBlocks(ctA, Lb, Box)
}

func TestBlocksC2PMul__Debug(t *testing.T) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	//Each step is compared with the plaintext pipeline of block matrix operations
	ADim := []int{4, 4}
	BDim := []int{4, 4}
	rowPA := 2
	colPA := 2
	rowPB := 2
	colPB := 2
	rd := rand.New(rand.NewSource(0))

	A := make([][]float64, ADim[0])
	for i := range A {
		A[i] = make([]float64, ADim[1])
		for j := range A[i] {
			A[i][j] = rd.NormFloat64()
		}
	}
	Ab, err := plainUtils.PartitionMatrix(plainUtils.NewDense(A), rowPA, colPA)

	B := make([][]float64, BDim[0])
	for i := range B {
		B[i] = make([]float64, BDim[1])
		for j := range B[i] {
			B[i][j] = rd.NormFloat64()
		}
	}
	Bb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(B), rowPB, colPB)

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
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

	rotations := GenRotations(Ab.InnerRows, 1, []int{Bb.InnerRows}, []int{Bb.InnerCols}, params, nil)

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
	X, err := NewEncInput(A, rowPA, colPA, Box)
	utils.ThrowErr(err)
	W, err := NewPlainWeightDiag(B, rowPB, colPB, X.InnerRows, Box)
	utils.ThrowErr(err)

	//BlOCK MULT ROUTINE

	if X.ColP != W.RowP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	q := X.RowP
	r := W.ColP
	s := W.RowP
	C := new(EncInput)
	Blocks := make([][]*mat.Dense, q)
	C.RowP = X.RowP
	C.ColP = W.ColP
	C.InnerRows = X.InnerRows
	C.InnerCols = W.InnerCols

	var wg sync.WaitGroup

	C.Blocks = make([][]*ckks.Ciphertext, q)
	for i := 0; i < q; i++ {
		C.Blocks[i] = make([]*ckks.Ciphertext, r)
		Blocks[i] = make([]*mat.Dense, r)
		for j := 0; j < r; j++ {
			partials := make([]*ckks.Ciphertext, s)
			partialsPlain := make([]*mat.Dense, s)
			for k := 0; k < s; k++ {
				//cipher
				wg.Add(1)
				x := X.Blocks[i][k]
				w := W.Blocks[k][j].Diags
				dimIn := X.InnerRows
				dimMid := W.InnerRows
				//dimOut := W.InnerCols
				//ckks.stuff are not thread safe -> recreate on the flight
				box := CkksBox{
					Params:    Box.Params,
					Encoder:   Box.Encoder.ShallowCopy(),
					Evaluator: Box.Evaluator.ShallowCopy(),
					Decryptor: nil,
					Encryptor: nil,
				}
				go func(x *ckks.Ciphertext, w []*ckks.Plaintext, dimIn, dimMid, k int, res []*ckks.Ciphertext, Box CkksBox) {
					defer wg.Done()
					res[k] = Cipher2PMul(x, dimIn, dimMid, w, true, true, Box)
				}(x, w, dimIn, dimMid, k, partials, box)
				//plain
				wg.Add(1)
				innerRows := Ab.InnerRows
				innerCols := Bb.InnerCols
				go func(a, b *mat.Dense, k int, res []*mat.Dense) {
					defer wg.Done()
					cij := mat.NewDense(innerRows, innerCols, nil)
					cij.Mul(a, b)
					res[k] = cij
				}(Ab.Blocks[i][k], Bb.Blocks[k][j], k, partialsPlain)
			}
			wg.Wait()
			Cij := partials[0]
			bij := partialsPlain[0]

			for k := 1; k < s; k++ {
				Cij = Box.Evaluator.AddNew(Cij, partials[k])
				bij.Add(bij, partialsPlain[k])
			}
			C.Blocks[i][j] = Cij
			Blocks[i][j] = bij
			//CompareMatrices(Cij, plainUtils.NumRows(bij), plainUtils.NumCols(bij), bij, Box)
		}
	}
	Cpb := &plainUtils.BMatrix{Blocks: Blocks, RowP: q, ColP: r, InnerRows: Ab.InnerRows, InnerCols: Bb.InnerCols}
	CompareBlocks(C, Cpb, Box)
}

func TestBlockCipher2P(t *testing.T) {
	LDim := []int{128, 841}
	W0Dim := []int{841, 845}
	W1Dim := []int{845, 100}
	W2Dim := []int{100, 10}

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

	W2 := make([][]float64, W2Dim[0])
	for i := range W2 {
		W2[i] = make([]float64, W2Dim[1])
		for j := range W2[i] {
			W2[i][j] = r.NormFloat64()
		}
	}

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 1, 29)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 29, 65)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 65, 10)
	W2b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W2), 10, 1)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.MultiPlyBlocks(B, W1b)
	utils.ThrowErr(err)
	D, err := plainUtils.MultiPlyBlocks(C, W2b)

	//ckksParams := bootstrapping.DefaultCKKSParameters[0]
	//ckksParams := ckks.PN14QP438
	//params, err := ckks.NewParametersFromLiteral(ckksParams)
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	utils.ThrowErr(err)

	//params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	//	LogN:         15,
	//	LogQ:         []int{60, 60, 60, 40, 40},
	//	LogP:         []int{61, 61},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     14,
	//	DefaultScale: float64(1 << 40),
	//})

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(len(L), 3, []int{W0b.InnerRows, W1b.InnerRows, W2b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols, W2b.InnerCols}, params, nil)

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

	ctA, err := NewEncInput(L, 1, 29, Box)
	utils.ThrowErr(err)
	W0bp, err := NewPlainWeightDiag(W0, 29, 65, ctA.InnerRows, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainWeightDiag(W1, 65, 10, ctA.InnerRows, Box)
	utils.ThrowErr(err)
	W2bp, err := NewPlainWeightDiag(W2, 10, 1, ctA.InnerRows, Box)
	utils.ThrowErr(err)
	fmt.Println("Start multiplications")
	now := time.Now()
	ctB, err := BlocksC2PMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 1", time.Since(now))
	CompareBlocks(ctB, B, Box)
	ctC, err := BlocksC2PMul(ctB, W1bp, Box)
	utils.ThrowErr(err)
	ctD, err := BlocksC2PMul(ctC, W2bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 2")
	CompareBlocks(ctD, D, Box)
}

func TestBlockCipher2C(t *testing.T) {
	LDim := []int{64, 64}
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

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 8, 16)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 16, 32)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 32, 16)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.MultiPlyBlocks(B, W1b)
	utils.ThrowErr(err)

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

	rotations := GenRotations(len(L), 2, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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

	ctA, err := NewEncInput(L, 8, 16, Box)
	utils.ThrowErr(err)
	W0bp, err := NewEncWeightDiag(W0, 16, 32, ctA.InnerRows, Box)
	utils.ThrowErr(err)
	W1bp, err := NewEncWeightDiag(W1, 32, 16, ctA.InnerRows, Box)
	utils.ThrowErr(err)

	ctB, err := BlocksC2CMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 1")
	CompareBlocks(ctB, B, Box)
	ctC, err := BlocksC2CMul(ctB, W1bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 2")
	CompareBlocks(ctC, C, Box)
}

func TestAddBlockCipher2P(t *testing.T) {
	LDim := []int{64, 845}
	W0Dim := []int{64, 845}
	W1Dim := []int{64, 845}

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

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 1, 65)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 1, 65)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 1, 65)

	B, err := plainUtils.AddBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.AddBlocks(B, W1b)
	utils.ThrowErr(err)

	//ckksParams := bootstrapping.DefaultCKKSParameters[0]
	//ckksParams := ckks.PN14QP438
	//params, err := ckks.NewParametersFromLiteral(ckksParams)
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	utils.ThrowErr(err)

	//params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	//	LogN:         15,
	//	LogQ:         []int{60, 60, 60, 40, 40},
	//	LogP:         []int{61, 61},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     14,
	//	DefaultScale: float64(1 << 40),
	//})

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(len(L), 2, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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

	ctA, err := NewEncInput(L, 1, 65, Box)
	utils.ThrowErr(err)
	W0bp, err := NewPlainInput(W0, 1, 65, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainInput(W1, 1, 65, Box)
	utils.ThrowErr(err)
	fmt.Println("Start additions")
	now := time.Now()
	ctB, err := AddBlocksC2P(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Add 1", time.Since(now))
	CompareBlocks(ctB, B, Box)
	ctC, err := AddBlocksC2P(ctB, W1bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Add 2")
	CompareBlocks(ctC, C, Box)
}

func TestMixCipher2P(t *testing.T) {
	// x then +
	LDim := []int{1, 841}
	W0Dim := []int{841, 845}
	W1Dim := []int{1, 845}

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

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 1, 29)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 29, 65)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 1, 65)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.AddBlocks(B, W1b)
	utils.ThrowErr(err)

	//ckksParams := bootstrapping.DefaultCKKSParameters[0]
	//ckksParams := ckks.PN14QP438
	//params, err := ckks.NewParametersFromLiteral(ckksParams)
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	utils.ThrowErr(err)

	//params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	//	LogN:         15,
	//	LogQ:         []int{60, 60, 60, 40, 40},
	//	LogP:         []int{61, 61},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     14,
	//	DefaultScale: float64(1 << 40),
	//})

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(len(L), 2, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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

	ctA, err := NewEncInput(L, 1, 29, Box)
	utils.ThrowErr(err)
	W0bp, err := NewPlainWeightDiag(W0, 29, 65, ctA.InnerRows, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainInput(W1, 1, 65, Box)
	utils.ThrowErr(err)
	now := time.Now()
	ctB, err := BlocksC2PMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 1", time.Since(now))
	CompareBlocks(ctB, B, Box)
	ctC, err := AddBlocksC2P(ctB, W1bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Add 2")
	CompareBlocks(ctC, C, Box)
}
