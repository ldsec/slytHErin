package cipherUtils

import (
	"errors"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

/*
OPERATIONS ON BLOCK MATRIXES (ENCRYPTED OR PLAIN WEIGHTS) DEFINED IN cipherUtils.encBlocks.go
For reference about the algorithms, check the plaintext equivalent in plainUtils.blocks.go
*/

func BlocksC2PMul(X *EncInput, W *PlainWeightDiag, Box CkksBox) (C *EncInput, err error) {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	err = nil
	if X.ColP != W.RowP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	q := X.RowP
	r := W.ColP
	s := W.RowP
	C = new(EncInput)
	C.RowP = X.RowP
	C.ColP = W.ColP
	C.InnerRows = X.InnerRows
	C.InnerCols = W.InnerCols

	var wg sync.WaitGroup

	C.Blocks = make([][]*ckks.Ciphertext, q)
	for i := 0; i < q; i++ {
		C.Blocks[i] = make([]*ckks.Ciphertext, r)
		for j := 0; j < r; j++ {
			partials := make([]*ckks.Ciphertext, s)
			for k := 0; k < s; k++ {
				wg.Add(1)
				x := X.Blocks[i][k]
				w := W.Blocks[k][j].Diags
				dimIn := X.InnerRows
				dimMid := W.InnerRows
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
			C.Blocks[i][j] = Cij
		}
	}
	return
}

func BlocksC2CMul(X *EncInput, W *EncWeightDiag, Box CkksBox) (C *EncInput, err error) {
	//multiplies 2 block matrices, both encrypted
	err = nil
	if X.ColP != W.RowP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	q := X.RowP
	r := W.ColP
	s := W.RowP
	C = new(EncInput)
	C.RowP = X.RowP
	C.ColP = W.ColP
	C.InnerRows = X.InnerRows
	C.InnerCols = W.InnerCols

	var wg sync.WaitGroup

	C.Blocks = make([][]*ckks.Ciphertext, q)
	for i := 0; i < q; i++ {
		C.Blocks[i] = make([]*ckks.Ciphertext, r)
		for j := 0; j < r; j++ {
			partials := make([]*ckks.Ciphertext, s)
			for k := 0; k < s; k++ {
				wg.Add(1)
				x := X.Blocks[i][k]
				w := W.Blocks[k][j].Diags
				dimIn := X.InnerRows
				dimMid := W.InnerRows
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
			C.Blocks[i][j] = Cij
		}
	}
	return
}

func AddBlocksC2C(A *EncInput, B *EncInput, Box CkksBox) (*EncInput, error) {
	var err error
	if A.RowP != B.RowP || A.ColP != B.ColP {
		err = errors.New("Block partitions not compatible for addition")
	}
	if A.InnerRows != B.InnerRows || A.InnerCols != B.InnerCols {
		err = errors.New("Inner dimensions not compatible for addition")
	}
	C := new(EncInput)
	C.RowP = A.RowP
	C.ColP = A.ColP
	C.InnerRows = C.InnerRows
	C.InnerCols = A.InnerCols
	C.Blocks = make([][]*ckks.Ciphertext, C.RowP)
	for i := range C.Blocks {
		C.Blocks[i] = make([]*ckks.Ciphertext, C.ColP)
		for j := range C.Blocks[0] {
			C.Blocks[i][j] = Box.Evaluator.AddNew(A.Blocks[i][j], B.Blocks[i][j])
		}
	}
	return C, err
}

func AddBlocksC2P(A *EncInput, B *PlainInput, Box CkksBox) (*EncInput, error) {
	var err error
	if A.RowP != B.RowP || A.ColP != B.ColP {
		err = errors.New("Block partitions not compatible for addition")
	}
	if A.InnerRows != B.InnerRows || A.InnerCols != B.InnerCols {
		err = errors.New("Inner dimensions not compatible for addition")
	}
	C := new(EncInput)
	C.RowP = A.RowP
	C.ColP = A.ColP
	C.InnerRows = C.InnerRows
	C.InnerCols = A.InnerCols
	C.Blocks = make([][]*ckks.Ciphertext, C.RowP)
	for i := range C.Blocks {
		C.Blocks[i] = make([]*ckks.Ciphertext, C.ColP)
		for j := range C.Blocks[0] {
			C.Blocks[i][j] = Box.Evaluator.AddNew(A.Blocks[i][j], B.Blocks[i][j])
		}
	}
	return C, err
}
