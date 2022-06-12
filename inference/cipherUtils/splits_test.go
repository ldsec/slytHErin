package cipherUtils

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"testing"
)

func TestFindSplits_SimpleNet(t *testing.T) {
	inputFeatures := 784
	weightRows := []int{784, 100}
	weightCols := []int{100, 10}
	ckksParams := ckks.ParametersLiteral{
		LogN:         13,
		LogQ:         []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 218 for LogN 13
		LogP:         []int{33},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     12,
		DefaultScale: float64(1 << 26),
	}

	params, _ := ckks.NewParametersFromLiteral(ckksParams)
	for _, strategy := range []bool{true, false} {
		fmt.Printf("\n\n\n")
		if strategy {
			fmt.Println("[!] Strategy on increasing Batch")
		} else {
			fmt.Println("[!] Strategy on tweaking weights split")
		}
		splits := FindSplits(inputFeatures, weightRows, weightCols, params, strategy)
		for i := range splits {
			fmt.Printf("\nPossible split %d\n", i+1)
			for j := range splits[i] {
				var splittingWhat string
				if j == 0 {
					splittingWhat = "Input"
				} else {
					splittingWhat = fmt.Sprintf("Weight %d", j)
				}
				split := splits[i][j]
				fmt.Println("Splits for ", splittingWhat)
				fmt.Printf("InR: %d InC: %d RP: %d CP: %d\n", split.InnerRows, split.InnerCols, split.RowP, split.ColP)
			}
		}
	}
}
