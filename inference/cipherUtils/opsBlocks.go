package cipherUtils

import "C"
import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"sync"
)

/*
OPERATIONS ON BLOCK MATRIXES (ENCRYPTED OR PLAIN) DEFINED IN cipherUtils.encBlocks.go
For reference about the algorithms, check the plaintext equivalent in plainUtils.blocks.go
*/

//Multiplication between encrypted input and plaintext weight
func BlocksC2PMul(X *EncInput, W *PlainWeightDiag, Box CkksBox) (*EncInput, error) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	var err error
	//W is block-transposed
	if X.ColP != W.ColP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	q := X.RowP
	//r and s are swapped cause W is block-transposed
	r := W.RowP
	s := W.ColP

	dimIn := X.InnerRows
	dimMid := W.InnerRows
	dimOut := W.InnerCols

	E := new(EncInput)
	E.RowP = X.RowP
	E.ColP = W.RowP
	E.InnerRows = X.InnerRows
	E.InnerCols = W.InnerCols
	E.Blocks = make([][]*ckks.Ciphertext, q)

	var wg sync.WaitGroup

	for i := 0; i < q; i++ {
		E.Blocks[i] = make([]*ckks.Ciphertext, r)
		Xrow := X.Blocks[i]
		for j := 0; j < r; j++ {
			Wcol := W.Blocks[j]
			wg.Add(1)
			go func(i, j int, Xrow []*ckks.Ciphertext, Wcol []*PlainDiagMat) {
				defer wg.Done()
				accumulatorChan := make(chan *ckks.Ciphertext, s-1)
				done := make(chan struct{})
				for k := 0; k < s; k++ {
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
						eij := Cipher2PMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
						if k == 0 {
							defer close(accumulatorChan)
							accumulator := 1
							for accumulator < s {
								op := <-accumulatorChan
								Box.Evaluator.Add(eij, op, eij)
								accumulator++
							}
							E.Blocks[i][j] = eij
							done <- struct{}{}
						} else {
							accumulatorChan <- eij
						}
					}(x, w, k, box)
				}
				<-done
			}(i, j, Xrow, Wcol)
		}
	}
	wg.Wait()
	utils.ThrowErr(err)
	return E, err
}

//Multiplication between encrypted input and weight
func BlocksC2CMul(X *EncInput, W *EncWeightDiag, Box CkksBox) (*EncInput, error) {
	//multiplies 2 block matrices, both encrypted
	var err error
	//W is block-transposed
	if X.ColP != W.ColP {
		err = errors.New("Block partitions not compatible for multiplication")
		utils.ThrowErr(err)
	}
	if X.InnerCols < W.InnerCols {
		err = errors.New("Dim Mid must be >= Dim Out in sub-matrices")
		utils.ThrowErr(err)
	}
	q := X.RowP
	//r and s swapped -> W is block tranposed
	r := W.RowP
	s := W.ColP

	dimIn := X.InnerRows
	dimMid := W.InnerRows
	dimOut := W.InnerCols

	E := new(EncInput)
	E.RowP = X.RowP
	E.ColP = W.RowP //W is block-transposed
	E.InnerRows = X.InnerRows
	E.InnerCols = W.InnerCols
	E.Blocks = make([][]*ckks.Ciphertext, q)

	var wg sync.WaitGroup

	for i := 0; i < q; i++ {
		E.Blocks[i] = make([]*ckks.Ciphertext, r)
		Xrow := X.Blocks[i]
		for j := 0; j < r; j++ {
			Wcol := W.Blocks[j]
			wg.Add(1)
			go func(i, j int, Xrow []*ckks.Ciphertext, Wcol []*EncDiagMat) {
				defer wg.Done()
				accumulatorChan := make(chan *ckks.Ciphertext, s-1)
				done := make(chan struct{})
				for k := 0; k < s; k++ {
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

					go func(x *ckks.Ciphertext, w []*ckks.Ciphertext, k int, Box CkksBox) {
						eij := Cipher2CMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
						if k == 0 {
							defer close(accumulatorChan)
							accumulator := 1
							for accumulator < s {
								op := <-accumulatorChan
								Box.Evaluator.Add(eij, op, eij)
								accumulator++
							}
							E.Blocks[i][j] = eij
							done <- struct{}{}
						} else {
							accumulatorChan <- eij
						}
					}(x, w, k, box)
				}
				<-done
			}(i, j, Xrow, Wcol)
		}
	}
	wg.Wait()
	utils.ThrowErr(err)
	return E, err
}

//Addition between encrypted input and weight
func AddBlocksC2C(A *EncInput, B *EncInput, Box CkksBox) (*EncInput, error) {
	var err error
	if A.RowP != B.RowP || A.ColP != B.ColP {
		err = errors.New("Block partitions not compatible for addition")
	}
	if A.InnerRows != B.InnerRows || A.InnerCols != B.InnerCols {
		err = errors.New("Inner dimensions not compatible for addition")
	}
	E := new(EncInput)
	E.RowP = A.RowP
	E.ColP = A.ColP
	E.InnerRows = A.InnerRows
	E.InnerCols = A.InnerCols
	E.Blocks = make([][]*ckks.Ciphertext, E.RowP)
	for i := range E.Blocks {
		E.Blocks[i] = make([]*ckks.Ciphertext, E.ColP)
		wg := sync.WaitGroup{}
		for j := range E.Blocks[0] {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				E.Blocks[i][j] = Box.Evaluator.AddNew(A.Blocks[i][j], B.Blocks[i][j])
			}(i, j)
		}
		wg.Wait()
	}
	utils.ThrowErr(err)
	return E, err
}

//Addition between encrypted input and plaintext weight
func AddBlocksC2P(A *EncInput, B *PlainInput, Box CkksBox) (*EncInput, error) {
	var err error
	if A.RowP != B.RowP || A.ColP != B.ColP {
		err = errors.New("Block partitions not compatible for addition")
	}
	if A.InnerRows != B.InnerRows || A.InnerCols != B.InnerCols {
		err = errors.New("Inner dimensions not compatible for addition")
	}
	E := new(EncInput)
	E.RowP = A.RowP
	E.ColP = A.ColP
	E.InnerRows = A.InnerRows
	E.InnerCols = A.InnerCols
	E.Blocks = make([][]*ckks.Ciphertext, E.RowP)
	for i := range E.Blocks {
		E.Blocks[i] = make([]*ckks.Ciphertext, E.ColP)
		wg := sync.WaitGroup{}
		for j := range E.Blocks[0] {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				E.Blocks[i][j] = Box.Evaluator.AddNew(A.Blocks[i][j], B.Blocks[i][j])
			}(i, j)
		}
		wg.Wait()
	}
	utils.ThrowErr(err)
	return E, err
}

//Evaluates a polynomial on the ciphertext
func EvalPolyBlocks(X *EncInput, poly *ckks.Polynomial, Box CkksBox) {
	//build map of all slots with legit values

	slotsIndex := make(map[int][]int)
	idx := make([]int, X.InnerRows*X.InnerCols)
	for i := 0; i < X.InnerRows*X.InnerCols; i++ {
		idx[i] = i
	}
	slotsIndex[0] = idx
	var wg sync.WaitGroup
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			wg.Add(1)
			go func(eval ckks.Evaluator, ecd ckks.Encoder, i, j int) {
				defer wg.Done()
				ct, _ := eval.EvaluatePolyVector(X.Blocks[i][j], []*ckks.Polynomial{poly}, ecd, slotsIndex, X.Blocks[i][j].Scale)
				X.Blocks[i][j] = ct
			}(Box.Evaluator.ShallowCopy(), Box.Encoder.ShallowCopy(), i, j)
		}
	}
	wg.Wait()
}

//Centralized Bootstrapping
func BootStrapBlocks(X *EncInput, Box CkksBox) {
	var wg sync.WaitGroup
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			wg.Add(1)
			go func(btp *bootstrapping.Bootstrapper, i, j int) {
				defer wg.Done()
				X.Blocks[i][j] = btp.Bootstrapp(X.Blocks[i][j])
			}(Box.BootStrapper.ShallowCopy(), i, j)
		}
	}
	wg.Wait()
	fmt.Println("Level after bootstrapping: ", X.Blocks[0][0].Level())
}

//Dummy Bootstrap where cipher is freshly encrypted
func DummyBootStrapBlocks(X *EncInput, Box CkksBox) *EncInput {
	pt := DecInput(X, Box)
	Xnew, err := NewEncInput(pt, X.RowP, X.ColP, Box.Params.MaxLevel(), Box)
	utils.ThrowErr(err)
	return Xnew
}

//Deprecated
func RescaleBlocks(X *EncInput, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			Box.Evaluator.Rescale(X.Blocks[i][j], Box.Params.DefaultScale(), X.Blocks[i][j])
		}
	}
}

//Deprecated
func RemoveImagFromBlocks(X *EncInput, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			Box.Evaluator.MultByConst(X.Blocks[i][j], 0.5, X.Blocks[i][j])
			Box.Evaluator.Add(X.Blocks[i][j], Box.Evaluator.ConjugateNew(X.Blocks[i][j]), X.Blocks[i][j])
			//Box.Evaluator.Rescale(X.Blocks[i][j], Box.Params.DefaultScale(), X.Blocks[i][j])
		}
	}
	RescaleBlocks(X, Box)
}
