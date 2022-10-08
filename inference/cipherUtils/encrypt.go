package cipherUtils

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
)

func EncryptInput(level int, scale float64, w [][]float64, Box CkksBox) *ckks.Ciphertext {
	params := Box.Params
	ecd := Box.Encoder
	enc := Box.Encryptor

	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, scale)
	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return enc.EncryptNew(pt)
}

func EncodeInput(level int, scale float64, w [][]float64, Box CkksBox) *ckks.Plaintext {
	params := Box.Params
	ecd := Box.Encoder

	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, scale)

	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return pt
}

//takes level, weight matrix, rows of input matrix to be multiplied, and box. Returns encrypted weight in diagonal form
func EncryptWeights(level int, w [][]float64, leftR, leftC int, Box CkksBox) *EncDiagMat {
	params := Box.Params
	ecd := Box.Encoder
	enc := Box.Encryptor

	wF := FormatWeights(w, leftR)
	ctW := make(map[int]*ckks.Ciphertext, len(wF))

	//for i := range ctW {
	// pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
	//	ecd.EncodeSlots(wF[i], pt, params.LogSlots())
	//	ctW[i] = enc.EncryptNew(pt)
	//}

	for i := range wF {
		pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ctW[i] = enc.EncryptNew(pt)
	}

	return &EncDiagMat{
		Diags:     ctW,
		InnerRows: len(w),
		InnerCols: len(w[0]),
		LeftR:     leftR,
		LeftC:     leftC,
		Encrypted: true,
	}
}

//takes level, weight matrix (square) rows of input matrix to be multiplied, and box. Returns plaintext weight in diagonal form
func EncodeWeights(level int, w [][]float64, leftR, leftC int, Box CkksBox) *PlainDiagMat {
	params := Box.Params
	ecd := Box.Encoder

	wF := FormatWeights(w, leftR)
	ctW := make(map[int]*ckks.Plaintext)

	//for i := range ctW {
	// pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
	//	ecd.EncodeSlots(wF[i], pt, params.LogSlots())
	//	ctW[i] = enc.EncryptNew(pt)
	//}
	for i := range wF {
		pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ctW[i] = pt
	}
	return &PlainDiagMat{
		Diags:     ctW,
		InnerRows: len(w),
		InnerCols: len(w[0]),
		LeftR:     leftR,
		LeftC:     leftC,
		Encrypted: false,
	}
}
