package nn

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"time"
)

/*
	|
	| MODEL IN CLEAR
	|
	V
*/

//Centralized
func (nne *NNEcd) EvalBatchEncrypted_Debug(Xenc *cipherUtils.EncInput, Y []int, Xclear *mat.Dense, weights, biases []*mat.Dense, activations []*utils.ChebyPolyApprox, labels int, Btp *cipherUtils.Bootstrapper) utils.Stats {
	Xint := Xenc
	XintPlain := Xclear
	Box := nne.Box

	now := time.Now()
	var prepack bool
	for i := range nne.Weights {
		W := nne.Weights[i]
		B := nne.Bias[i]

		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level == 0 {
			fmt.Println("Level 0, Bootstrapping...")
			fmt.Println("pre boot")
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			Btp.Bootstrap(Xint)
			fmt.Println("after boot")
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}
		if i == 0 {
			prepack = false
			Xint = nne.Multiplier.Multiply(Xint, W, prepack)
		} else {
			prepack = true
			Xint = nne.Multiplier.Multiply(Xint, W, prepack)
		}

		var tmp mat.Dense
		tmp.Mul(XintPlain, weights[i])
		tmpBlocks, err := plainUtils.PartitionMatrix(&tmp, Xint.RowP, Xint.ColP)
		utils.ThrowErr(err)
		fmt.Printf("Mul ")

		//bias
		nne.Adder.AddBias(Xint, B)
		utils.ThrowErr(err)

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpBlocks, err = plainUtils.PartitionMatrix(&tmp2, Xint.RowP, Xint.ColP)
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 0.9, Box)

		//activation
		if i != len(nne.Weights)-1 {
			level = Xint.Blocks[0][0].Level()
			if level < nne.ReLUApprox[i].LevelsOfAct() {
				fmt.Println("Bootstrapping for Activation")
				fmt.Println("pre boot")
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
				Btp.Bootstrap(Xint)
				fmt.Println("after boot")
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)

			}
			fmt.Println("Activation ")

			nne.Activators[i].ActivateBlocks(Xint)

			utils.ActivatePlain(&tmp2, activations[i])
			XintPlain = &tmp2
			XintPlainBlocks, _ := plainUtils.PartitionMatrix(XintPlain, Xint.RowP, Xint.ColP)
			cipherUtils.PrintDebugBlocks(Xint, XintPlainBlocks, 0.9, Box)
		} else {
			XintPlain = &tmp2
		}
	}
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	res := cipherUtils.DecInput(Xint, Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)
	correctsP, accuracyP, _ := utils.Predict(Y, labels, plainUtils.MatToArray(XintPlain))
	fmt.Println("Corrects: Enc/Plain :", corrects, " / ", correctsP)
	fmt.Println("Accuracy: Enc/Plain :", accuracy, " / ", accuracyP)
	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        elapsed,
	}
}

//Centralized
func (nne *NNEcd) EvalBatchEncrypted(Xenc *cipherUtils.EncInput, Y []int, labels int, Btp *cipherUtils.Bootstrapper) utils.Stats {
	Xint := Xenc

	now := time.Now()

	var prepack bool

	for i := range nne.Weights {
		W := nne.Weights[i]
		B := nne.Bias[i]

		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level == 0 {
			fmt.Println("Level 0, Bootstrapping...")
			Btp.Bootstrap(Xint)
			fmt.Println("after boot")
			fmt.Println(Xint.Blocks[0][0].Scale - nne.Box.Params.DefaultScale())
			fmt.Println("Level: ", Xint.Blocks[0][0].Level())
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}
		if i == 0 {
			prepack = false
			Xint = nne.Multiplier.Multiply(Xint, W, prepack)
		} else {
			prepack = true
			Xint = nne.Multiplier.Multiply(Xint, W, prepack)
		}
		fmt.Println(Xint.Blocks[0][0].Scale - nne.Box.Params.DefaultScale())

		//bias
		nne.Adder.AddBias(Xint, B)
		fmt.Println(Xint.Blocks[0][0].Scale - nne.Box.Params.DefaultScale())
		fmt.Println("Level: ", Xint.Blocks[0][0].Level())

		//activation
		if i != len(nne.Weights)-1 {
			level = Xint.Blocks[0][0].Level()
			if level < nne.ReLUApprox[i].LevelsOfAct() {
				fmt.Println("Bootstrapping for Activation")
				fmt.Println("pre boot")
				fmt.Println("Level: ", Xint.Blocks[0][0].Level())
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
				Btp.Bootstrap(Xint)
				fmt.Println("after boot")
				fmt.Println("Level: ", Xint.Blocks[0][0].Level())
				fmt.Println(Xint.Blocks[0][0].Scale - nne.Box.Params.DefaultScale())
			}
			fmt.Println("Activation ")

			nne.Activators[i].ActivateBlocks(Xint)
			fmt.Println("Level: ", Xint.Blocks[0][0].Level())
			fmt.Println(Xint.Blocks[0][0].Scale - nne.Box.Params.DefaultScale())
		}
	}
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	res := cipherUtils.DecInput(Xint, nne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)
	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        elapsed,
	}
}
