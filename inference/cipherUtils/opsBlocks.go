package cipherUtils

import "C"
import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

/*
OPERATIONS ON BLOCK MATRIXES (ENCRYPTED OR PLAIN) DEFINED IN cipherUtils.encBlocks.go
For reference about the algorithms, check the plaintext equivalent in plainUtils.blocks.go
*/

func BlocksC2PMul(X *EncInput, W *PlainWeightDiag, Box CkksBox) (*EncInput, error) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	var err error
	if X.ColP != W.RowP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	q := X.RowP
	r := W.ColP
	s := W.RowP
	E := new(EncInput)
	E.RowP = X.RowP
	E.ColP = W.ColP
	E.InnerRows = X.InnerRows
	E.InnerCols = W.InnerCols

	var wg sync.WaitGroup

	E.Blocks = make([][]*ckks.Ciphertext, q)
	for i := 0; i < q; i++ {
		E.Blocks[i] = make([]*ckks.Ciphertext, r)
		for j := 0; j < r; j++ {
			partials := make([]*ckks.Ciphertext, s)
			for k := 0; k < s; k++ {
				wg.Add(1)
				x := X.Blocks[i][k]
				w := W.Blocks[k][j].Diags
				dimIn := X.InnerRows
				dimMid := W.InnerRows
				//dimOut := W.InnerCols
				//ckks.stuff not thread-safe -> recreate on flight
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
			}
			wg.Wait()
			Cij := partials[0]
			for k := 1; k < s; k++ {
				Cij = Box.Evaluator.AddNew(Cij, partials[k])
			}
			E.Blocks[i][j] = Cij
		}
	}
	utils.ThrowErr(err)
	return E, err
}

func BlocksC2CMul(X *EncInput, W *EncWeightDiag, Box CkksBox) (*EncInput, error) {
	//multiplies 2 block matrices, both encrypted
	var err error
	if X.ColP != W.RowP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	q := X.RowP
	r := W.ColP
	s := W.RowP
	E := new(EncInput)
	E.RowP = X.RowP
	E.ColP = W.ColP
	E.InnerRows = X.InnerRows
	E.InnerCols = W.InnerCols

	var wg sync.WaitGroup

	E.Blocks = make([][]*ckks.Ciphertext, q)
	for i := 0; i < q; i++ {
		E.Blocks[i] = make([]*ckks.Ciphertext, r)
		for j := 0; j < r; j++ {
			partials := make([]*ckks.Ciphertext, s)
			for k := 0; k < s; k++ {
				wg.Add(1)
				x := X.Blocks[i][k]
				w := W.Blocks[k][j].Diags
				dimIn := X.InnerRows
				dimMid := W.InnerRows
				//dimOut := W.InnerCols
				//ckks.stuff not thread-safe -> recreate on flight
				box := CkksBox{
					Params:    Box.Params,
					Encoder:   Box.Encoder.ShallowCopy(),
					Evaluator: Box.Evaluator.ShallowCopy(),
					Decryptor: nil,
					Encryptor: nil,
				}
				go func(x *ckks.Ciphertext, w []*ckks.Ciphertext, dimIn, dimMid, k int, res []*ckks.Ciphertext, Box CkksBox) {
					defer wg.Done()
					cij := Cipher2CMul(x, dimIn, dimMid, w, true, true, Box)
					res[k] = cij
				}(x, w, dimIn, dimMid, k, partials, box)
			}
			wg.Wait()
			Cij := partials[0]
			for k := 1; k < s; k++ {
				Cij = Box.Evaluator.AddNew(Cij, partials[k])
			}
			E.Blocks[i][j] = Cij
		}
	}
	utils.ThrowErr(err)
	return E, err
}

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
		for j := range E.Blocks[0] {
			E.Blocks[i][j] = Box.Evaluator.AddNew(A.Blocks[i][j], B.Blocks[i][j])
		}
	}
	utils.ThrowErr(err)
	return E, err
}

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
		for j := range E.Blocks[0] {
			E.Blocks[i][j] = Box.Evaluator.AddNew(A.Blocks[i][j], B.Blocks[i][j])
		}
	}
	utils.ThrowErr(err)
	return E, err
}

func EvalPolyBlocks(X *EncInput, coeffs []float64, Box CkksBox) {
	poly := ckks.NewPoly(plainUtils.RealToComplex(coeffs))
	//fmt.Println("Deg", poly.Degree())
	//fmt.Println("Level before:", X.Blocks[0][0].Level())

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
	//fmt.Println("Level after:", X.Blocks[0][0].Level())
}

func BootStrapBlocks(X *EncInput, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			X.Blocks[i][j] = Box.BootStrapper.Bootstrapp(X.Blocks[i][j])
		}
	}
}

func RescaleBlocks(X *EncInput, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			Box.Evaluator.Rescale(X.Blocks[i][j], Box.Params.DefaultScale(), X.Blocks[i][j])
		}
	}
}

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
