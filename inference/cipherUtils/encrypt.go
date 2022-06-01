package cipherUtils

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

func EncryptInput(level int, w [][]float64, Box CkksBox) *ckks.Ciphertext {
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
