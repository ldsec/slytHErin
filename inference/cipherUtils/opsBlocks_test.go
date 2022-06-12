package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math"
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

var LDim = []int{1, 784}
var W0Dim = []int{784, 100}
var W1Dim = []int{100, 10}

var L = plainUtils.MatToArray(plainUtils.RandMatrix(LDim[0], LDim[1]))
var W0 = plainUtils.MatToArray(plainUtils.RandMatrix(W0Dim[0], W0Dim[1]))
var W1 = plainUtils.MatToArray(plainUtils.RandMatrix(W1Dim[0], W1Dim[1]))

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
	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 2, 2)

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

	ctA, err := NewEncInput(L, 1, 1, params.MaxLevel(), Box)
	CompareBlocks(ctA, Lb, Box)
}

func TestBlocksC2PMul__Debug(t *testing.T) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	//Each step is compared with the plaintext pipeline of block matrix operations
	rowP := []int{1}
	ADim := []int{156, 676}
	BDim := []int{676, 92}

	rd := rand.New(rand.NewSource(0))

	A := make([][]float64, ADim[0])
	for i := range A {
		A[i] = make([]float64, ADim[1])
		for j := range A[i] {
			A[i][j] = rd.NormFloat64()
		}
	}
	B := make([][]float64, BDim[0])
	for i := range B {
		B[i] = make([]float64, BDim[1])
		for j := range B[i] {
			B[i][j] = rd.NormFloat64()
		}
	}
	for _, rp := range rowP {

		rowPA := rp
		colPA := 26
		rowPB := 26
		colPB := 4

		Ab, err := plainUtils.PartitionMatrix(plainUtils.NewDense(A), rowPA, colPA)

		Bb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(B), rowPB, colPB)

		params, err := ckks.NewParametersFromLiteral(ckks.PN14QP438)
		if err != nil {
			panic(err)
		}

		kgen := ckks.NewKeyGenerator(params)
		sk := kgen.GenSecretKey()
		rlk := kgen.GenRelinearizationKey(sk, 2)

		rotations := GenRotations(Ab.InnerRows, Ab.InnerCols, 1, []int{Bb.InnerRows}, []int{Bb.InnerCols}, params, nil)

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
		X, err := NewEncInput(A, rowPA, colPA, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		W, err := NewPlainWeightDiag(B, rowPB, colPB, X.InnerRows, params.MaxLevel(), Box)
		utils.ThrowErr(err)

		//BlOCK MULT ROUTINE
		now := time.Now()

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
					//copy the values
					x := X.Blocks[i][k].CopyNew()
					w := W.Blocks[k][j].Diags
					dimIn := X.InnerRows
					dimMid := W.InnerRows
					dimOut := W.InnerCols
					//fmt.Println("Dims:", dimIn, " ", dimMid, " ", dimOut)
					//ckks.stuff are not thread safe -> recreate on the flight
					box := CkksBox{
						Params:    Box.Params,
						Encoder:   Box.Encoder.ShallowCopy(),
						Evaluator: Box.Evaluator.ShallowCopy(),
						Decryptor: nil,
						Encryptor: nil,
					}
					go func(x *ckks.Ciphertext, w []*ckks.Plaintext, dimIn, dimMid, dimOut, k int, res []*ckks.Ciphertext, Box CkksBox) {
						defer wg.Done()
						res[k] = Cipher2PMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
					}(x, w, dimIn, dimMid, dimOut, k, partials, box)
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
				//fmt.Println("k: ", 0)
				//PrintDebug(Cij, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(bij))), Box)
				for k := 1; k < s; k++ {
					//fmt.Println("k: ", k)
					//PrintDebug(partials[k], plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(partialsPlain[k]))), Box)
					Cij = Box.Evaluator.AddNew(Cij, partials[k])
					bij.Add(bij, partialsPlain[k])
				}
				C.Blocks[i][j] = Cij
				Blocks[i][j] = bij
				//fmt.Println(i, "  ", j, "__________________________________")
				//CompareMatrices(Cij, plainUtils.NumRows(bij), plainUtils.NumCols(bij), bij, Box)
				//PrintDebug(Cij, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(bij))), Box)
			}
		}
		Cpb := &plainUtils.BMatrix{Blocks: Blocks, RowP: q, ColP: r, InnerRows: Ab.InnerRows, InnerCols: Bb.InnerCols}
		fmt.Println("Done: ", time.Since(now))
		CompareBlocks(C, Cpb, Box)
		PrintDebugBlocks(C, Cpb, Box)

	}
}

func TestBlocksC2PMul_Parallel_Debug(t *testing.T) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	//Each step is compared with the plaintext pipeline of block matrix operations
	//This code uses NxLxM threads if blocks are NxL X LxM
	rowP := []int{1}
	ADim := []int{64, 676}
	BDim := []int{676, 92}

	rd := rand.New(rand.NewSource(0))

	A := make([][]float64, ADim[0])
	for i := range A {
		A[i] = make([]float64, ADim[1])
		for j := range A[i] {
			A[i][j] = rd.NormFloat64()
		}
	}
	B := make([][]float64, BDim[0])
	for i := range B {
		B[i] = make([]float64, BDim[1])
		for j := range B[i] {
			B[i][j] = rd.NormFloat64()
		}
	}
	for _, rp := range rowP {

		rowPA := rp
		colPA := 26
		rowPB := 26
		colPB := 4

		Ab, err := plainUtils.PartitionMatrix(plainUtils.NewDense(A), rowPA, colPA)

		Bb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(B), rowPB, colPB)

		params, err := ckks.NewParametersFromLiteral(ckks.PN14QP438)
		if err != nil {
			panic(err)
		}

		kgen := ckks.NewKeyGenerator(params)
		sk := kgen.GenSecretKey()
		rlk := kgen.GenRelinearizationKey(sk, 2)

		rotations := GenRotations(Ab.InnerRows, Ab.InnerCols, 1, []int{Bb.InnerRows}, []int{Bb.InnerCols}, params, nil)

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
		X, err := NewEncInput(A, rowPA, colPA, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		W, err := NewPlainWeightDiag(B, rowPB, colPB, X.InnerRows, params.MaxLevel(), Box)
		utils.ThrowErr(err)

		//BlOCK MULT ROUTINE
		now := time.Now()

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

		var wgi sync.WaitGroup

		C.Blocks = make([][]*ckks.Ciphertext, q)
		for i := 0; i < q; i++ {
			C.Blocks[i] = make([]*ckks.Ciphertext, r)
			Blocks[i] = make([]*mat.Dense, r)
			wgi.Add(1)
			go func(i int) {
				defer wgi.Done()
				var wgj sync.WaitGroup
				for j := 0; j < r; j++ {
					partials := make([]*ckks.Ciphertext, s)
					partialsPlain := make([]*mat.Dense, s)
					wgj.Add(1)
					go func(i, j int) {
						defer wgj.Done()
						var wgk sync.WaitGroup
						for k := 0; k < s; k++ {
							//cipher
							wgk.Add(1)
							x := X.Blocks[i][k].CopyNew()
							w := W.Blocks[k][j].Diags
							dimIn := X.InnerRows
							dimMid := W.InnerRows
							dimOut := W.InnerCols
							//ckks.stuff are not thread safe -> recreate on the flight
							box := CkksBox{
								Params:    Box.Params,
								Encoder:   Box.Encoder.ShallowCopy(),
								Evaluator: Box.Evaluator.ShallowCopy(),
								Decryptor: nil,
								Encryptor: nil,
							}
							go func(x *ckks.Ciphertext, w []*ckks.Plaintext, dimIn, dimMid, dimOut, k int, res []*ckks.Ciphertext, Box CkksBox) {
								defer wgk.Done()
								res[k] = Cipher2PMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
							}(x, w, dimIn, dimMid, dimOut, k, partials, box)
							//plain
							wgk.Add(1)
							innerRows := Ab.InnerRows
							innerCols := Bb.InnerCols
							go func(a, b *mat.Dense, k int, res []*mat.Dense) {
								defer wgk.Done()
								cij := mat.NewDense(innerRows, innerCols, nil)
								cij.Mul(a, b)
								res[k] = cij
							}(Ab.Blocks[i][k], Bb.Blocks[k][j], k, partialsPlain)
						}
						wgk.Wait()
						Cij := partials[0]
						bij := partialsPlain[0]
						for k := 1; k < s; k++ {
							Cij = Box.Evaluator.AddNew(Cij, partials[k])
							bij.Add(bij, partialsPlain[k])
						}
						C.Blocks[i][j] = Cij
						Blocks[i][j] = bij
						//CompareMatrices(Cij, plainUtils.NumRows(bij), plainUtils.NumCols(bij), bij, Box)
					}(i, j)
				}
				wgj.Wait()
			}(i)
		}
		wgi.Wait()
		Cpb := &plainUtils.BMatrix{Blocks: Blocks, RowP: q, ColP: r, InnerRows: Ab.InnerRows, InnerCols: Bb.InnerCols}
		fmt.Println("Done: ", time.Since(now))
		CompareBlocks(C, Cpb, Box)
		PrintDebugBlocks(C, Cpb, Box)
	}
}

func TestBlocksC2PMul_Parallel_Accumulator_Debug(t *testing.T) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	//Each step is compared with the plaintext pipeline of block matrix operations
	//Adds accumulator to threads
	rowP := []int{1}
	ADim := []int{64, 676}
	BDim := []int{676, 92}

	rd := rand.New(rand.NewSource(0))

	A := make([][]float64, ADim[0])
	for i := range A {
		A[i] = make([]float64, ADim[1])
		for j := range A[i] {
			A[i][j] = rd.NormFloat64()
		}
	}
	B := make([][]float64, BDim[0])
	for i := range B {
		B[i] = make([]float64, BDim[1])
		for j := range B[i] {
			B[i][j] = rd.NormFloat64()
		}
	}
	for _, rp := range rowP {

		rowPA := rp
		colPA := 26
		rowPB := 26
		colPB := 4

		Ab, err := plainUtils.PartitionMatrix(plainUtils.NewDense(A), rowPA, colPA)

		Bb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(B), rowPB, colPB)

		params, err := ckks.NewParametersFromLiteral(ckks.PN14QP438)
		if err != nil {
			panic(err)
		}

		kgen := ckks.NewKeyGenerator(params)
		sk := kgen.GenSecretKey()
		rlk := kgen.GenRelinearizationKey(sk, 2)

		rotations := GenRotations(Ab.InnerRows, Ab.InnerCols, 1, []int{Bb.InnerRows}, []int{Bb.InnerCols}, params, nil)

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
		X, err := NewEncInput(A, rowPA, colPA, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		W, err := NewPlainWeightDiag(B, rowPB, colPB, X.InnerRows, params.MaxLevel(), Box)
		utils.ThrowErr(err)

		//BlOCK MULT ROUTINE
		now := time.Now()

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

		dimIn := X.InnerRows
		dimMid := W.InnerRows
		dimOut := W.InnerCols

		innerRows := Ab.InnerRows
		innerCols := Bb.InnerCols

		var wgi sync.WaitGroup

		C.Blocks = make([][]*ckks.Ciphertext, q)
		for i := 0; i < q; i++ {
			C.Blocks[i] = make([]*ckks.Ciphertext, r)
			Blocks[i] = make([]*mat.Dense, r)
			wgi.Add(1)
			go func(i int) {
				defer wgi.Done()
				var wgj sync.WaitGroup
				for j := 0; j < r; j++ {

					accumulatorChan := make(chan *ckks.Ciphertext)
					accumulatorPlainChan := make(chan *mat.Dense)
					done := make(chan struct{})
					donePlain := make(chan struct{})
					wgj.Add(1)

					go func(j int, accumulatorChan chan *ckks.Ciphertext, accumulatorChanPlain chan *mat.Dense, done, donePlain chan struct{}) {
						defer wgj.Done()
						for k := 0; k < s; k++ {
							//cipher
							x := X.Blocks[i][k].CopyNew()
							w := W.Blocks[k][j].Diags

							//ckks.stuff are not thread safe -> recreate on the flight
							box := CkksBox{
								Params:    Box.Params,
								Encoder:   Box.Encoder.ShallowCopy(),
								Evaluator: Box.Evaluator.ShallowCopy(),
								Decryptor: nil,
								Encryptor: nil,
							}

							go func(x *ckks.Ciphertext, w []*ckks.Plaintext, k int, Box CkksBox) {
								cij := Cipher2PMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
								if k == 0 {
									defer close(accumulatorChan)
									accumulator := 1

									for accumulator < s {
										op := <-accumulatorChan

										Box.Evaluator.Add(cij, op, cij)
										accumulator++

									}

									C.Blocks[i][j] = cij
									done <- struct{}{}

								} else {

									accumulatorChan <- cij
								}
							}(x, w, k, box)

							//plain
							go func(a, b *mat.Dense, k int) {
								cij := mat.NewDense(innerRows, innerCols, nil)
								cij.Mul(a, b)
								if k == 0 {
									defer close(accumulatorPlainChan)
									accumulator := 1
									for accumulator < s {
										op := <-accumulatorPlainChan
										cij.Add(cij, op)
										accumulator++
									}
									Blocks[i][j] = cij
									donePlain <- struct{}{}
								} else {
									accumulatorPlainChan <- cij
								}
							}(Ab.Blocks[i][k], Bb.Blocks[k][j], k)
						}
						<-done
						<-donePlain
						//CompareMatrices(Cij, plainUtils.NumRows(bij), plainUtils.NumCols(bij), bij, Box)
					}(j, accumulatorChan, accumulatorPlainChan, done, donePlain)
				}
				wgj.Wait()
			}(i)
		}
		wgi.Wait()
		Cpb := &plainUtils.BMatrix{Blocks: Blocks, RowP: q, ColP: r, InnerRows: Ab.InnerRows, InnerCols: Bb.InnerCols}
		fmt.Println("Done: ", time.Since(now))
		CompareBlocks(C, Cpb, Box)
		PrintDebugBlocks(C, Cpb, Box)
	}
}

func TestBlocksC2PMul_Parallel_Accumulator_Transposed_Debug(t *testing.T) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	//Each step is compared with the plaintext pipeline of block matrix operations
	//In this version matrix B blocks are transposed to leverage caching
	rowP := []int{2}
	ADim := []int{64, 676}
	BDim := []int{676, 92}

	rd := rand.New(rand.NewSource(0))

	A := make([][]float64, ADim[0])
	for i := range A {
		A[i] = make([]float64, ADim[1])
		for j := range A[i] {
			A[i][j] = rd.NormFloat64()
		}
	}
	B := make([][]float64, BDim[0])
	for i := range B {
		B[i] = make([]float64, BDim[1])
		for j := range B[i] {
			B[i][j] = rd.NormFloat64()
		}
	}
	for _, rp := range rowP {

		rowPA := rp
		colPA := 26
		rowPB := 26
		colPB := 4

		Ab, err := plainUtils.PartitionMatrix(plainUtils.NewDense(A), rowPA, colPA)

		Bb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(B), rowPB, colPB)
		Bt := plainUtils.TransposeBlocks(Bb)
		params, err := ckks.NewParametersFromLiteral(ckks.PN14QP438)
		if err != nil {
			panic(err)
		}

		kgen := ckks.NewKeyGenerator(params)
		sk := kgen.GenSecretKey()
		rlk := kgen.GenRelinearizationKey(sk, 2)

		rotations := GenRotations(Ab.InnerRows, Ab.InnerCols, 1, []int{Bb.InnerRows}, []int{Bb.InnerCols}, params, nil)

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
		X, err := NewEncInput(A, rowPA, colPA, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		W, err := NewPlainWeightDiag(plainUtils.DenseToMatrix(plainUtils.ExpandBlocks(Bt)), Bt.RowP, Bt.ColP, X.InnerRows, params.MaxLevel(), Box)
		utils.ThrowErr(err)

		//BlOCK MULT ROUTINE
		now := time.Now()

		if X.ColP != W.RowP {
			err = errors.New("Block partitions not compatible for multiplication")
		}
		q := X.RowP
		r := W.RowP
		s := W.ColP
		C := new(EncInput)
		Blocks := make([][]*mat.Dense, q)
		C.RowP = X.RowP
		C.ColP = W.RowP
		C.InnerRows = X.InnerRows
		C.InnerCols = W.InnerCols

		dimIn := X.InnerRows
		dimMid := W.InnerRows
		dimOut := W.InnerCols

		innerRows := Ab.InnerRows
		innerCols := Bb.InnerCols

		var wgi sync.WaitGroup

		C.Blocks = make([][]*ckks.Ciphertext, q)
		for i := 0; i < q; i++ {
			C.Blocks[i] = make([]*ckks.Ciphertext, r)
			Blocks[i] = make([]*mat.Dense, r)
			Xrow := X.Blocks[i]
			wgi.Add(1)
			go func(i int, Xrow []*ckks.Ciphertext) {
				defer wgi.Done()
				var wgj sync.WaitGroup
				for j := 0; j < r; j++ {

					accumulatorChan := make(chan *ckks.Ciphertext)
					accumulatorPlainChan := make(chan *mat.Dense)
					done := make(chan struct{})
					donePlain := make(chan struct{})
					Wcol := W.Blocks[j]
					wgj.Add(1)

					go func(j int, Wcol []*PlainDiagMat, accumulatorChan chan *ckks.Ciphertext, accumulatorChanPlain chan *mat.Dense, done, donePlain chan struct{}) {
						defer wgj.Done()
						for k := 0; k < s; k++ {
							//cipher
							x := Xrow[k].CopyNew()
							w := Wcol[k].Diags

							//ckks.stuff are not thread safe -> recreate on the flight
							box := CkksBox{
								Params:    Box.Params,
								Encoder:   Box.Encoder.ShallowCopy(),
								Evaluator: Box.Evaluator.ShallowCopy(),
								Decryptor: nil,
								Encryptor: nil,
							}

							go func(x *ckks.Ciphertext, w []*ckks.Plaintext, k int, Box CkksBox) {
								cij := Cipher2PMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
								if k == 0 {
									defer close(accumulatorChan)
									accumulator := 1

									for accumulator < s {
										op := <-accumulatorChan

										Box.Evaluator.Add(cij, op, cij)
										accumulator++

									}

									C.Blocks[i][j] = cij
									done <- struct{}{}

								} else {

									accumulatorChan <- cij
								}
							}(x, w, k, box)

							//plain
							go func(a, b *mat.Dense, k int) {
								cij := mat.NewDense(innerRows, innerCols, nil)
								cij.Mul(a, b)
								if k == 0 {
									defer close(accumulatorPlainChan)
									accumulator := 1
									for accumulator < s {
										op := <-accumulatorPlainChan
										cij.Add(cij, op)
										accumulator++
									}
									Blocks[i][j] = cij
									donePlain <- struct{}{}
								} else {
									accumulatorPlainChan <- cij
								}
							}(Ab.Blocks[i][k], Bb.Blocks[k][j], k)
						}
						<-done
						<-donePlain
						//CompareMatrices(Cij, plainUtils.NumRows(bij), plainUtils.NumCols(bij), bij, Box)
					}(j, Wcol, accumulatorChan, accumulatorPlainChan, done, donePlain)
				}
				wgj.Wait()
			}(i, Xrow)
		}
		wgi.Wait()
		Cpb := &plainUtils.BMatrix{Blocks: Blocks, RowP: q, ColP: r, InnerRows: Ab.InnerRows, InnerCols: Bb.InnerCols}
		fmt.Println("Done: ", time.Since(now))
		CompareBlocks(C, Cpb, Box)
		PrintDebugBlocks(C, Cpb, Box)
	}
}

func TestBlockCipher2P(t *testing.T) {
	fmt.Println("Encrypted to Plain")
	rowP := 1
	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), rowP, 49)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 49, 1)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 1, 1)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.MultiPlyBlocks(B, W1b)
	utils.ThrowErr(err)

	ckksParams := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	utils.ThrowErr(err)

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(Lb.InnerRows, Lb.InnerCols, 2, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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
	level := params.MaxLevel()
	ctA, err := NewEncInput(L, rowP, 49, level, Box)
	utils.ThrowErr(err)
	W0bp, err := NewPlainWeightDiag(W0, 49, 1, ctA.InnerRows, level, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainWeightDiag(W1, 1, 1, ctA.InnerRows, level-1, Box)
	utils.ThrowErr(err)

	utils.ThrowErr(err)
	fmt.Println("Start multiplications")
	now := time.Now()
	ctB, err := BlocksC2PMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 1 ", time.Since(now))
	PrintDebugBlocks(ctB, B, Box)
	now = time.Now()
	ctC, err := BlocksC2PMul(ctB, W1bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 2 ", time.Since(now))
	PrintDebugBlocks(ctC, C, Box)

}

func TestBlockCipher2C(t *testing.T) {
	fmt.Println("Encrypted to Encrypted")
	rowP := 1
	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), rowP, 29)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 29, 65)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 65, 10)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.MultiPlyBlocks(B, W1b)
	utils.ThrowErr(err)
	utils.ThrowErr(err)
	ckksParams := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	if err != nil {
		panic(err)
	}

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(Lb.InnerRows, Lb.InnerCols, 3, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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
	level := params.MaxLevel()
	ctA, err := NewEncInput(L, Lb.RowP, 29, level, Box)
	utils.ThrowErr(err)
	W0bp, err := NewEncWeightDiag(W0, 29, 65, ctA.InnerRows, level, Box)
	utils.ThrowErr(err)
	W1bp, err := NewEncWeightDiag(W1, 65, 10, ctA.InnerRows, level-1, Box)
	utils.ThrowErr(err)

	ctB, err := BlocksC2CMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 1")

	PrintDebugBlocks(ctB, B, Box)
	ctC, err := BlocksC2CMul(ctB, W1bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul 2")

	PrintDebugBlocks(ctC, C, Box)

}

func TestBlockCipherMul(t *testing.T) {
	//make sure they have the same params
	t.Run("E2P", TestBlockCipher2P)
	t.Run("E2C", TestBlockCipher2C)
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

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 1, 10)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 10, 1)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 1, 1)

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

	rotations := GenRotations(len(L), len(L[0]), 2, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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
	level := params.MaxLevel()
	ctA, err := NewEncInput(L, 1, 65, level, Box)
	utils.ThrowErr(err)
	W0bp, err := NewPlainInput(W0, 1, 65, level, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainInput(W1, 1, 65, level, Box)
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
	// x then + and activation
	LDim := []int{64, 100}
	W0Dim := []int{100, 10}
	W1Dim := []int{64, 10}

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
	coeffs := []float64{1.1155, 5.0, 4.4003} //degree 2
	interval := 10.0                         //--> incorporate this in weight matrix to spare a level

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 2, 10)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 10, 1)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 2, 1)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.AddBlocks(B, W1b)
	utils.ThrowErr(err)

	for i := 0; i < C.RowP; i++ {
		for j := 0; j < C.ColP; j++ {
			func(X *mat.Dense, interval float64, degree int, coeffs []float64) {
				rows, cols := X.Dims()
				for r := 0; r < rows; r++ {
					for c := 0; c < cols; c++ {
						v := X.At(r, c) / float64(interval)
						res := 0.0
						for deg := 0; deg < degree; deg++ {
							res += (math.Pow(v, float64(deg)) * coeffs[deg])
						}
						X.Set(r, c, res)
					}
				}
			}(C.Blocks[i][j], interval, 3, coeffs)
		}
	}
	//ckksParams := bootstrapping.DefaultCKKSParameters[0]
	ckksParams := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(ckksParams)

	utils.ThrowErr(err)

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(Lb.InnerRows, Lb.InnerCols, 1, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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
	level := params.MaxLevel()
	ctA, err := NewEncInput(L, 2, 10, level, Box)
	utils.ThrowErr(err)
	for i := range W0 {
		for j := range W0[i] {
			W0[i][j] *= 1 / interval
		}
	}

	for i := range W1 {
		for j := range W1[i] {
			W1[i][j] *= 1 / interval
		}
	}
	W0bp, err := NewPlainWeightDiag(W0, 10, 1, ctA.InnerRows, level, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainInput(W1, 2, 1, level-1, Box)
	utils.ThrowErr(err)
	now := time.Now()
	ctB, err := BlocksC2PMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul", time.Since(now))
	fmt.Println("Add")
	ctC, err := AddBlocksC2P(ctB, W1bp, Box)
	fmt.Println("Activation")
	EvalPolyBlocks(ctC, ckks.NewPoly(plainUtils.RealToComplex(coeffs)), Box)
	fmt.Println("Done...", time.Since(now))
	utils.ThrowErr(err)
	CompareBlocks(ctC, C, Box)
	PrintDebugBlocks(ctC, C, Box)
}

func TestActivator(t *testing.T) {
	// x then + and activation
	LDim := []int{64, 100}
	W0Dim := []int{100, 10} //weight
	W1Dim := []int{64, 10}  //bias

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
	//relu approx
	f := func(x float64) float64 {
		return math.Log(1 + math.Exp(x))
	}
	a, b := -30.0, 30.0
	deg := 31
	approxF := ckks.Approximate(f, a, b, deg)

	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 2, 10)
	utils.ThrowErr(err)
	W0b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W0), 10, 1)
	utils.ThrowErr(err)
	W1b, err := plainUtils.PartitionMatrix(plainUtils.NewDense(W1), 2, 1)

	B, err := plainUtils.MultiPlyBlocks(Lb, W0b)
	utils.ThrowErr(err)
	C, err := plainUtils.AddBlocks(B, W1b)
	utils.ThrowErr(err)

	plainUtils.ApplyFunc(C, f)
	//ckksParams := bootstrapping.DefaultCKKSParameters[0]
	ckksParams := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(ckksParams)

	utils.ThrowErr(err)

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := GenRotations(Lb.InnerRows, Lb.InnerCols, 1, []int{W0b.InnerRows, W1b.InnerRows}, []int{W0b.InnerCols, W1b.InnerCols}, params, nil)

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
	level := params.MaxLevel()
	ctA, err := NewEncInput(L, 2, 10, level, Box)
	utils.ThrowErr(err)
	for i := range W0 {
		for j := range W0[i] {
			W0[i][j] *= 2 / (b - a)
		}
	}

	for i := range W1 {
		for j := range W1[i] {
			W1[i][j] *= 2 / (b - a)
			W1[i][j] += (-a - b) / (b - a)
		}
	}
	act, err := NewActivator(approxF, level-1, params.DefaultScale(), ctA.InnerRows, W0b.InnerCols, ctA.RowP, W0b.ColP, Box)
	utils.ThrowErr(err)
	W0bp, err := NewPlainWeightDiag(W0, 10, 1, ctA.InnerRows, level, Box)
	utils.ThrowErr(err)
	W1bp, err := NewPlainInput(W1, 2, 1, level-1, Box)
	utils.ThrowErr(err)
	ctB, err := BlocksC2PMul(ctA, W0bp, Box)
	utils.ThrowErr(err)
	fmt.Println("Mul")
	fmt.Println("Add")
	ctC, err := AddBlocksC2P(ctB, W1bp, Box)
	fmt.Println("Activation")
	now := time.Now()
	act.ActivateBlocks(ctC)
	fmt.Println("Done...", time.Since(now))
	utils.ThrowErr(err)
	CompareBlocks(ctC, C, Box)
	PrintDebugBlocks(ctC, C, Box)
}

func TestRemoveImagFromBlocks(t *testing.T) {
	LDim := []int{4, 4}
	r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = r.NormFloat64()
		}
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	utils.ThrowErr(err)
	Lb, err := plainUtils.PartitionMatrix(plainUtils.NewDense(L), 2, 2)
	utils.ThrowErr(err)
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)
	rotations := GenRotations(len(L), len(L[0]), 1, []int{Lb.InnerRows, Lb.InnerRows}, []int{Lb.InnerCols, Lb.InnerCols}, params, nil)
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
	ct, err := NewEncInput(L, 2, 2, params.MaxLevel(), Box)
	utils.ThrowErr(err)
	RemoveImagFromBlocks(ct, Box)

	CompareBlocks(ct, Lb, Box)
	PrintDebugBlocks(ct, Lb, Box)
}
