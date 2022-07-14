package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"math"
)

type DebugStats struct {
	MinPrec  float64
	AvgPrec  float64
	MaxPrec  float64
	MaxValue float64
	L2Dist   float64
}

func PrintDebug(ciphertext *ckks.Ciphertext, valuesWant []complex128, thresh float64, Box CkksBox) DebugStats {
	encoder := Box.Encoder
	params := Box.Params
	decryptor := Box.Decryptor

	valuesTest := encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())[:len(valuesWant)]

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

func PrintDebugBlocks(X *EncInput, Pt *plainUtils.BMatrix, thresh float64, Box CkksBox) {
	fmt.Println("[?] Debug Info:-------------------------------------------------------------------------")
	stats := DebugStats{}
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			//because the plaintext in X.Blocks is the matrix transposed and flattened, transpose the plaintext
			var ptm *mat.Dense
			ptm = plainUtils.TransposeDense(Pt.Blocks[i][j])

			pt := plainUtils.MatToArray(ptm)
			stat := PrintDebug(X.Blocks[i][j], plainUtils.RealToComplex(plainUtils.Vectorize(pt, true)), thresh, Box)
			stats.MaxPrec += stat.MaxPrec
			stats.MinPrec += stat.MinPrec
			stats.AvgPrec += stat.AvgPrec
			if stats.MaxValue < stat.MaxValue {
				stats.MaxValue = stat.MaxValue
			}
			stats.L2Dist += stat.L2Dist
		}
	}
	stats.MaxPrec /= float64(X.RowP * X.ColP)
	stats.MinPrec /= float64(X.RowP * X.ColP)
	stats.AvgPrec /= float64(X.RowP * X.ColP)
	stats.L2Dist /= float64(X.RowP * X.ColP)

	fmt.Println("[!] Final Stats:")
	fmt.Printf("MAX Prec: %f\n", stats.MaxPrec)
	fmt.Printf("MIN Prec: %f\n", stats.MinPrec)
	fmt.Printf("AVG Prec: %f\n", stats.AvgPrec)
	fmt.Printf("MAX Value: %f\n", stats.MaxValue)
	fmt.Printf("L2 Dist: %f\n", stats.L2Dist)
	stats.MaxPrec /= float64(X.RowP * X.ColP)
	stats.MinPrec /= float64(X.RowP * X.ColP)
	stats.AvgPrec /= float64(X.RowP * X.ColP)
	stats.L2Dist /= float64(X.RowP * X.ColP)

	params := Box.Params

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", X.Blocks[0][0].Level(), params.LogQLvl(X.Blocks[0][0].Level()))
	fmt.Println("Consumed levels:", params.MaxLevel()-X.Blocks[0][0].Level())
	fmt.Println("Difference with Scale: ", X.Blocks[0][0].Scale-params.DefaultScale())

	fmt.Println("[!] Final Stats:")
	fmt.Printf("MAX Prec: %f\n", stats.MaxPrec)
	fmt.Printf("MIN Prec: %f\n", stats.MinPrec)
	fmt.Printf("AVG Prec: %f\n", stats.AvgPrec)
	fmt.Printf("MAX Value: %f\n", stats.MaxValue)
	fmt.Printf("L2 Dist: %f\n", stats.L2Dist)

	fmt.Println("----------------------------------------------------------------------------------------")
	return

	fmt.Println("----------------------------------------------------------------------------------------")
	return
}
