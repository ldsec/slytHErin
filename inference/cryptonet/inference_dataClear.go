package cryptonet

import (
	"fmt"
	cU "github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"time"
)

func (cne *CryptonetEnc) EvalBatchWithModelEnc(X *cU.PlainInput, Y []int, labels int) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	iAct := 0
	var prepack bool
	res := new(cU.EncInput)
	for i := range cne.Weights {
		if i == 0 {
			prepack = false
			res = cne.Multiplier.Multiply(X, cne.Weights[i], prepack)
		} else {
			prepack = true
			res = cne.Multiplier.Multiply(res, cne.Weights[i], prepack)
		}

		cne.Adder.AddBias(res, cne.Bias[i])

		if iAct < 2 {
			cne.Activators[iAct].ActivateBlocks(res)
			iAct++
		}
	}
	//client masks its result
	mask := cU.DecodeInput(cU.MaskInput(res, cne.Box), cne.Box)

	//server decrypts
	resP := cU.DecInput(res, cne.Box)

	//unmask (we could do this generating a secret key at the server and a switching key to switch from server key to this ephemeral key and then make the client decrypt using the ephemral key)
	for i := range resP {
		for j := range resP[i] {
			resP[i][j] -= mask[i][j]
		}
	}

	end := time.Since(start)
	fmt.Println("Done ", end)

	corrects, accuracy, predictions := utils.Predict(Y, labels, resP)

	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        end,
	}
}

func (cne *CryptonetEnc) EvalBatchWithModelEnc_Debug(X *cU.PlainInput, Xclear *mat.Dense, weights, biases []*mat.Dense, activation *utils.ChebyPolyApprox, Y []int, labels int) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	iAct := 0
	var prepack bool
	res := new(cU.EncInput)
	for i := range cne.Weights {
		if i == 0 {
			prepack = false
			res = cne.Multiplier.Multiply(X, cne.Weights[i], prepack)
		} else {
			prepack = true
			res = cne.Multiplier.Multiply(res, cne.Weights[i], prepack)
		}

		var tmp mat.Dense
		tmp.Mul(Xclear, weights[i])
		tmpB, _ := plainUtils.PartitionMatrix(&tmp, res.RowP, res.ColP)
		cU.PrintDebugBlocks(res, tmpB, 0.1, cne.Box)

		cne.Adder.AddBias(res, cne.Bias[i])

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpB, _ = plainUtils.PartitionMatrix(&tmp2, res.RowP, res.ColP)
		cU.PrintDebugBlocks(res, tmpB, 0.1, cne.Box)

		if iAct < 2 {
			cne.Activators[iAct].ActivateBlocks(res)
			utils.ActivatePlain(&tmp2, activation)
			*Xclear = tmp2
			tmpB, _ = plainUtils.PartitionMatrix(&tmp2, res.RowP, res.ColP)
			cU.PrintDebugBlocks(res, tmpB, 1, cne.Box)
		}
		iAct++
		*Xclear = tmp2
	}
	//client masks its result
	//mask := cU.DecodeInput(cU.MaskInput(res, cne.Box), cne.Box)
	//
	////server decrypts
	//resP := cU.DecInput(res, cne.Box)
	//
	//for i := range resP {
	//	for j := range resP[i] {
	//		resP[i][j] -= mask[i][j]
	//	}
	//}
	fmt.Println("Level before masked decryption:", res.Blocks[0][0].Level())

	mask := cU.MaskInputV2(res, cne.Box, 128)

	//server decrypts
	resP := cU.DecInputNoDecode(res, cne.Box)

	//unmask
	cU.UnmaskInput(resP, mask, cne.Box)
	end := time.Since(start)
	fmt.Println("Done ", end)
	//resPlain := cU.DecInput(res, cne.Box)
	resPlain := cU.DecodeInput(resP, cne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, resPlain)

	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        end,
	}
}
