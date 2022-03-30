package cipherUtils

import (
	"errors"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

func BlocksC2PMul(X *EncInput, W *PlainWeight, Box CkksBox) (C *EncInput, err error) {
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
				go func(x *ckks.Ciphertext, w []*ckks.Plaintext, dimIn, dimMid, k int, res []*ckks.Ciphertext, Box CkksBox) {
					defer wg.Done()
					cij := Cipher2PMul(x, dimIn, dimMid, w, true, true, Box)
					res[k] = cij
				}(x, w, dimIn, dimMid, k, partials, Box)
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

func BlocksC2CMul(X *EncInput, W *EncWeight, Box CkksBox) (C *EncInput, err error) {
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
				go func(x *ckks.Ciphertext, w []*ckks.Ciphertext, dimIn, dimMid, k int, res []*ckks.Ciphertext, Box CkksBox) {
					defer wg.Done()
					cij := Cipher2CMul(x, dimIn, dimMid, w, true, true, Box)
					res[k] = cij
				}(x, w, dimIn, dimMid, k, partials, Box)
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
