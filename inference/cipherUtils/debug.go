package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
)

//Statistics for debug
type DebugStats struct {
	MinPrec  float64
	AvgPrec  float64
	MaxPrec  float64
	MaxValue float64
	L2Dist   float64
}

//Debug stats for one ciphertext
func PrintDebug(ciphertext *ckks.Ciphertext, valuesWant []complex128, thresh float64, Box CkksBox) DebugStats {
	encoder := Box.Encoder
	params := Box.Params
	decryptor := Box.Decryptor

	valuesTest := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())[:len(valuesWant)]

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)
	fmt.Println(precStats.String())

	dist := plainUtils.Distance(plainUtils.ComplexToReal(valuesTest), plainUtils.ComplexToReal(valuesWant))

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

	//compare L1 distance between values
	for i := range valuesWant {
		if math.Abs(real(valuesWant[i]-valuesTest[i])) > thresh {
			panic(errors.New(fmt.Sprintf("Expected %f, got %f, at %d", valuesWant[i], valuesTest[i], i)))
		}
	}
	return DebugStats{
		MinPrec:  precStats.MinPrecision.Real,
		AvgPrec:  precStats.MeanPrecision.Real,
		MaxPrec:  precStats.MaxPrecision.Real,
		MaxValue: maxTest,
		L2Dist:   dist,
	}
}

//Debug stats for block matrix
func PrintDebugBlocks(Xenc *EncInput, Pt *plainUtils.BMatrix, thresh float64, Box CkksBox) {
	fmt.Println("[?] Debug Info:-------------------------------------------------------------------------")

	X := plainUtils.Vectorize(DecInput(Xenc, Box), false)
	Y := plainUtils.Vectorize(plainUtils.MatToArray(plainUtils.ExpandBlocks(Pt)), false)

	precStats := ckks.GetPrecisionStats(Box.Params, Box.Encoder, nil, Y[:Box.Params.Slots()], X[:Box.Params.Slots()], Box.Params.LogSlots(), 0)
	fmt.Println(precStats.String())

	dist := plainUtils.Distance(X, Y)

	maxTest := 0.0
	maxWant := 0.0

	for i := 0; i < len(X); i++ {
		if math.Abs(X[i]) > maxTest {
			maxTest = math.Abs(X[i])
		}
		if math.Abs(Y[i]) > maxWant {
			maxWant = math.Abs(Y[i])
		}
	}

	//compare L1 distance between values
	for i := range Y {
		if math.Abs(Y[i]-X[i]) > thresh {
			panic(errors.New(fmt.Sprintf("Expected %f, got %f, at %d", Y[i], X[i], i)))
		}
	}
	fmt.Println(precStats.String())
	fmt.Println("Distance:", dist)
	fmt.Println("Max Test:", maxTest)
	fmt.Println("Max Test:", maxWant)
}
