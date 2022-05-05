package cipherUtils

import (
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"sync"
)

func EncryptInput(level int, w [][]float64, Box CkksBox) (ctW *ckks.Ciphertext) {
	params := Box.Params
	ecd := Box.Encoder
	enc := Box.Encryptor

	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, params.DefaultScale())
	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return enc.EncryptNew(pt)
}

func DecInput(XEnc *EncInput, Box CkksBox) [][]float64 {
	/*
		Given a block input matrix, decrypts and returns the underlying original matrix
		The sub-matrices are also transposed (remember that they are in form flatten(A.T))
	*/
	Xb := new(plainUtils.BMatrix)
	Xb.RowP = XEnc.RowP
	Xb.ColP = XEnc.ColP
	Xb.InnerRows = XEnc.InnerRows
	Xb.InnerCols = XEnc.InnerCols
	Xb.Blocks = make([][]*mat.Dense, Xb.RowP)
	for i := 0; i < XEnc.RowP; i++ {
		Xb.Blocks[i] = make([]*mat.Dense, Xb.ColP)
		for j := 0; j < XEnc.ColP; j++ {
			pt := Box.Decryptor.DecryptNew(XEnc.Blocks[i][j])
			ptArray := Box.Encoder.DecodeSlots(pt, Box.Params.LogSlots())
			//this is flatten(x.T)
			resReal := plainUtils.ComplexToReal(ptArray)[:XEnc.InnerRows*XEnc.InnerCols]
			res := plainUtils.TransposeDense(mat.NewDense(XEnc.InnerCols, XEnc.InnerRows, resReal))
			// now this is x
			Xb.Blocks[i][j] = res
		}
	}
	return plainUtils.MatToArray(plainUtils.ExpandBlocks(Xb))
}

func EncodeInput(level int, w [][]float64, Box CkksBox) *ckks.Plaintext {
	params := Box.Params
	ecd := Box.Encoder

	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, params.DefaultScale())

	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return pt
}

func EncryptWeights(level int, w [][]float64, leftdim int, Box CkksBox) (ctW []*ckks.Ciphertext) {
	params := Box.Params
	ecd := Box.Encoder
	enc := Box.Encryptor

	wF := FormatWeights(w, leftdim)

	ctW = make([]*ckks.Ciphertext, len(wF))

	//for i := range ctW {
	//	ecd.EncodeSlots(wF[i], pt, params.LogSlots())
	//	ctW[i] = enc.EncryptNew(pt)
	//}
	var wg sync.WaitGroup
	for i := range ctW {
		wg.Add(1)
		go func(i int, ecd ckks.Encoder, enc ckks.Encryptor) {
			defer wg.Done()
			pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
			ecd.EncodeSlots(wF[i], pt, params.LogSlots())
			ctW[i] = enc.EncryptNew(pt)
		}(i, ecd.ShallowCopy(), enc.ShallowCopy())
	}
	wg.Wait()

	return
}

func EncodeWeights(level int, w [][]float64, leftdim int, Box CkksBox) (ptW []*ckks.Plaintext) {
	params := Box.Params
	ecd := Box.Encoder

	wF := FormatWeights(w, leftdim)

	ptW = make([]*ckks.Plaintext, len(wF))
	var wg sync.WaitGroup
	for i := range ptW {
		wg.Add(1)
		go func(i int, ecd ckks.Encoder) {
			defer wg.Done()
			pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
			ecd.EncodeSlots(wF[i], pt, params.LogSlots())
			ptW[i] = pt
		}(i, ecd.ShallowCopy())
	}
	wg.Wait()
	return
}
