package cipherUtils

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
)

func EncryptInput(level int, w [][]float64, params ckks.Parameters, ecd ckks.Encoder, enc ckks.Encryptor) (ctW *ckks.Ciphertext) {
	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, params.DefaultScale())
	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return enc.EncryptNew(pt)
}

func EncryptWeights(level int, w [][]float64, leftdim int, params ckks.Parameters, ecd ckks.Encoder, enc ckks.Encryptor) (ctW []*ckks.Ciphertext) {
	wF := FormatWeights(w, leftdim)
	pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))

	ctW = make([]*ckks.Ciphertext, len(wF))

	for i := range ctW {
		//EncodeSlots puts the values in the plaintext in a way
		//such that homomorphic elem-wise mult is preserved
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ctW[i] = enc.EncryptNew(pt)
	}

	return
}

func PrintDebug(ciphertext *ckks.Ciphertext, valuesWant []complex128, Box CkksBox) (valuesTest []complex128) {
	encoder := Box.Encoder
	params := Box.Params
	decryptor := Box.Decryptor

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest: %6.10f %6.10f %6.10f %6.10f...\n", valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
	fmt.Printf("ValuesWant: %6.10f %6.10f %6.10f %6.10f...\n", valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}
