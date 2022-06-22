package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"math"
)

func PrintDebug(ciphertext *ckks.Ciphertext, valuesWant []complex128, thresh float64, Box CkksBox) (valuesTest []complex128) {
	fmt.Println("[?] Debug Info:-------------------------------------------------------------------------")

	encoder := Box.Encoder
	params := Box.Params
	decryptor := Box.Decryptor

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())[:len(valuesWant)]

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Println("Consumed levels:", params.MaxLevel()-ciphertext.Level())
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest:")
	for i := range valuesWant {
		fmt.Printf(" %6.10f", valuesTest[i])
	}
	fmt.Println()
	fmt.Printf("ValuesWant:")
	for i := range valuesWant {
		fmt.Printf(" %6.10f", valuesWant[i])
	}
	fmt.Println()

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println("L2 Distance:")
	fmt.Println(plainUtils.Distance(plainUtils.ComplexToReal(valuesTest), plainUtils.ComplexToReal(valuesWant)))
	fmt.Println()
	fmt.Println("Scale:")
	fmt.Println(Box.Params.DefaultScale() - ciphertext.Scale)
	fmt.Println()
	fmt.Println("Max value:")
	maxTest := 0.0
	maxWant := 0.0
	vT, vW := plainUtils.ComplexToReal(valuesTest), plainUtils.ComplexToReal(valuesWant)
	for i := 0; i < len(valuesWant); i++ {
		if math.Abs(vT[i]) > maxTest {
			maxTest = math.Abs(vT[i])
		}
		if math.Abs(vW[i]) > maxWant {
			maxWant = math.Abs(vW[i])
		}
	}
	fmt.Printf("Test: %.8f\n", maxTest)
	fmt.Printf("Want: %.8f\n\n", maxWant)
	for i := range valuesWant {
		if math.Abs(real(valuesWant[i]-valuesTest[i])) > thresh {
			panic(errors.New(fmt.Sprintf("Expected %f, got %f, at %d", valuesWant[i], valuesTest[i], i)))
		}
	}
	fmt.Println("----------------------------------------------------------------------------------------")
	return
}

func PrintDebugBlocks(X *EncInput, Pt *plainUtils.BMatrix, afterMul bool, thresh float64, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			//because the plaintext in X.Blocks is the matrix transposed and flattened, transpose the plaintext
			var ptm *mat.Dense
			if afterMul {
				ptm = plainUtils.TransposeDense(Pt.Blocks[i][j])
			} else {
				ptm = Pt.Blocks[i][j]
			}
			pt := plainUtils.MatToArray(ptm)
			PrintDebug(X.Blocks[i][j], plainUtils.RealToComplex(plainUtils.Vectorize(pt, true)), thresh, Box)
			return //only first
		}
	}

	return
}

//L2 distance between blocks of ct and pt
func CompareBlocksL2(Ct *EncInput, Pt *plainUtils.BMatrix, Box CkksBox) {
	ct := DecInput(Ct, Box)
	pt := plainUtils.MatToArray(plainUtils.ExpandBlocks(Pt))
	fmt.Println("Distance:", plainUtils.Distance(plainUtils.Vectorize(ct, true), plainUtils.Vectorize(pt, true)))
}

//L2 distance between pt and ct
func CompareMatricesL2(Ct *ckks.Ciphertext, rows, cols int, Pt *mat.Dense, Box CkksBox) {
	ct := Box.Decryptor.DecryptNew(Ct)
	ptArray := Box.Encoder.DecodeSlots(ct, Box.Params.LogSlots())
	//this is flatten(x.T)
	resReal := plainUtils.ComplexToReal(ptArray)[:rows*cols]
	res := plainUtils.TransposeDense(mat.NewDense(cols, rows, resReal))
	fmt.Println("Distance:",
		plainUtils.Distance(plainUtils.Vectorize(plainUtils.MatToArray(res), true),
			plainUtils.Vectorize(plainUtils.MatToArray(Pt), true)))
}
