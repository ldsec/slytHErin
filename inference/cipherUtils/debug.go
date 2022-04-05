package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"math"
)

func PrintDebug(ciphertext *ckks.Ciphertext, valuesWant []complex128, Box CkksBox) (valuesTest []complex128) {
	encoder := Box.Encoder
	params := Box.Params
	decryptor := Box.Decryptor

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())[:len(valuesWant)]

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

func CompareBlocks(Ct *EncInput, Pt *plainUtils.BMatrix, Box CkksBox) {
	ct := DecInput(Ct, Box)
	pt := plainUtils.MatToArray(plainUtils.ExpandBlocks(Pt))
	//fmt.Println("Dec:")
	//fmt.Println(ct)
	//fmt.Println("Expected:")
	//fmt.Println(pt)
	fmt.Println("Distance:", plainUtils.Distance(plainUtils.Vectorize(ct, true), plainUtils.Vectorize(pt, true)))
}

func CompareMatrices(Ct *ckks.Ciphertext, rows, cols int, Pt *mat.Dense, Box CkksBox) {
	ct := Box.Decryptor.DecryptNew(Ct)
	ptArray := Box.Encoder.DecodeSlots(ct, Box.Params.LogSlots())
	//this is flatten(x.T)
	resReal := plainUtils.ComplexToReal(ptArray)[:rows*cols]
	res := plainUtils.TransposeDense(mat.NewDense(cols, rows, resReal))
	fmt.Println("Distance:",
		plainUtils.Distance(plainUtils.Vectorize(plainUtils.MatToArray(res), true),
			plainUtils.Vectorize(plainUtils.MatToArray(Pt), true)))
}
