package cipherUtils

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
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

	pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))

	ctW = make([]*ckks.Ciphertext, len(wF))

	for i := range ctW {
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ctW[i] = enc.EncryptNew(pt)
	}

	return
}

func EncodeWeights(level int, w [][]float64, leftdim int, Box CkksBox) (ptW []*ckks.Plaintext) {
	params := Box.Params
	ecd := Box.Encoder

	wF := FormatWeights(w, leftdim)

	ptW = make([]*ckks.Plaintext, len(wF))

	for i := range ptW {
		//pt is a pointer -> we need a new one each time
		pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ptW[i] = pt
	}

	return
}
