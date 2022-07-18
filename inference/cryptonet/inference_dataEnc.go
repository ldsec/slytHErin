package cryptonet

import (
	"fmt"
	cU "github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"time"
)

func (cne *CryptonetEcd) EvalBatchEncrypted(Xenc *cU.EncInput, Y []int, labels int) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	iAct := 0
	var prepack bool
	for i := range cne.Weights {
		if i == 0 {
			prepack = false
		} else {
			prepack = true
		}
		Xenc = cne.Multiplier.Multiply(Xenc, cne.Weights[i], prepack)

		cne.Adder.AddBias(Xenc, cne.Bias[i])

		if iAct < 2 {
			cne.Activators[iAct].ActivateBlocks(Xenc)
			iAct++
		}
	}
	end := time.Since(start)
	fmt.Println("Done ", end)
	res := cU.DecInput(Xenc, cne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)

	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        end,
	}
}

func (cne *CryptonetEcd) EvalBatchEncrypted_Debug(Xenc *cU.EncInput, Xclear *mat.Dense, weights, biases []*mat.Dense, activation *utils.ChebyPolyApprox, Y []int, labels int) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	iAct := 0
	var prepack bool
	for i := range cne.Weights {
		if i == 0 {
			prepack = false
		} else {
			prepack = true
		}
		timer := time.Now()
		Xenc = cne.Multiplier.Multiply(Xenc, cne.Weights[i], prepack)
		finish := time.Since(timer)

		var tmp mat.Dense
		tmp.Mul(Xclear, weights[i])
		tmpB, _ := plainUtils.PartitionMatrix(&tmp, Xenc.RowP, Xenc.ColP)

		cne.Adder.AddBias(Xenc, cne.Bias[i])

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpB, _ = plainUtils.PartitionMatrix(&tmp2, Xenc.RowP, Xenc.ColP)
		cU.PrintDebugBlocks(Xenc, tmpB, 0.1, cne.Box)

		fmt.Println("Mul layer ", i+1, ": ", finish)

		if iAct < 2 {
			timer = time.Now()
			cne.Activators[iAct].ActivateBlocks(Xenc)
			finish = time.Since(timer)
			utils.ActivatePlain(&tmp2, activation)
			*Xclear = tmp2
			tmpB, _ = plainUtils.PartitionMatrix(&tmp2, Xenc.RowP, Xenc.ColP)
			cU.PrintDebugBlocks(Xenc, tmpB, 1, cne.Box)
			fmt.Println("Act layer ", i+1, ": ", finish)
		}
		iAct++
		*Xclear = tmp2
	}
	end := time.Since(start)
	fmt.Println("Done ", end)
	res := cU.DecInput(Xenc, cne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)

	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        end,
	}
}
