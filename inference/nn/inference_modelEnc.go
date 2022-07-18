package nn

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"time"
)

/*
	|
	| MODEL ENCRYPTED
	|
	V
*/
func (nne *NNEnc) EvalBatchEncrypted_Debug(Xenc *cipherUtils.EncInput, Y []int, Xclear *mat.Dense, weights, biases []*mat.Dense, activations []*utils.ChebyPolyApprox, labels int, Btp *cipherUtils.Bootstrapper) utils.Stats {
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
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 0.1, Box)

		//bias
		nne.Adder.AddBias(Xint, B)
		utils.ThrowErr(err)

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpBlocks, err = plainUtils.PartitionMatrix(&tmp2, Xint.RowP, Xint.ColP)

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
			fmt.Printf("Activation ")

			nne.Activators[i].ActivateBlocks(Xint)

			utils.ActivatePlain(&tmp2, activations[i])
			XintPlain = &tmp2
			XintPlainBlocks, _ := plainUtils.PartitionMatrix(XintPlain, Xint.RowP, Xint.ColP)
			cipherUtils.PrintDebugBlocks(Xint, XintPlainBlocks, 0.1, Box)
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

func (nne *NNEnc) EvalBatchEncrypted(Xenc *cipherUtils.EncInput, Y []int, labels int, Btp *cipherUtils.Bootstrapper) utils.Stats {
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

		//bias
		nne.Adder.AddBias(Xint, B)

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
			fmt.Printf("Activation ")

			nne.Activators[i].ActivateBlocks(Xint)
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

func (nne *NNEnc) EvalBatchEncrypted_Distributed_Debug(Xenc *cipherUtils.EncInput, Y []int, Xclear *mat.Dense, weights, biases []*mat.Dense, activations []*utils.ChebyPolyApprox, labels int, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int,
	master *distributed.LocalMaster) utils.Stats {

	Xint := Xenc
	XintPlain := Xclear

	now := time.Now()

	var prepack bool
	for i := range nne.Weights {
		W := nne.Weights[i]
		B := nne.Bias[i]

		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level <= minLevel && level < nne.LevelsToComplete(i, false) { //minLevel for Bootstrapping
			if level < minLevel {
				utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
			}
			fmt.Println("MinLevel, Bootstrapping...")
			master.StartProto(distributed.REFRESH, Xint, pkQ, minLevel)
			fmt.Println("Level after bootstrapping: ", Xint.Blocks[0][0].Level())
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
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 10, nne.Box)

		//bias
		nne.Adder.AddBias(Xint, B)

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpBlocks, err = plainUtils.PartitionMatrix(&tmp2, Xint.RowP, Xint.ColP)
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 10, nne.Box)

		level = Xint.Blocks[0][0].Level()
		if i != len(nne.Weights)-1 {
			if (level < nne.ReLUApprox[i].LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox[i].LevelsOfAct() < minLevel) && level < nne.LevelsToComplete(i, true) {
				if level < minLevel {
					utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
				}
				if level < nne.ReLUApprox[i].LevelsOfAct() {
					fmt.Printf("Level < %d before activation , Bootstrapping...\n", nne.ReLUApprox[i].LevelsOfAct())
				} else if level == minLevel {
					fmt.Println("Min Level , Bootstrapping...")
				} else {
					fmt.Println("Activation would set level below threshold, Pre-emptive Bootstraping...")
					fmt.Println("Curr level: ", level)
					fmt.Println("Drop to: ", minLevel)
					fmt.Println("Diff: ", level-minLevel)
				}
				master.StartProto(distributed.REFRESH, Xint, pkQ, minLevel)
			}
			fmt.Println("Activation")
			//activation
			nne.Activators[i].ActivateBlocks(Xint)
			utils.ActivatePlain(&tmp2, activations[i])
			XintPlain = &tmp2
			XintPlainBlocks, _ := plainUtils.PartitionMatrix(XintPlain, Xint.RowP, Xint.ColP)
			cipherUtils.PrintDebugBlocks(Xint, XintPlainBlocks, 10, nne.Box)
		} else {
			XintPlain = &tmp2
		}
	}
	fmt.Println("Key Switch to querier public key")
	master.StartProto(distributed.CKSWITCH, Xint, pkQ, minLevel)

	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	Box := nne.Box
	Box.Decryptor = decQ
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

func (nne *NNEnc) EvalBatchEncrypted_Distributed(Xenc *cipherUtils.EncInput, Y []int, labels int, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int,
	master *distributed.LocalMaster) utils.Stats {

	Xint := Xenc

	now := time.Now()
	var prepack bool

	for i := range nne.Weights {
		W := nne.Weights[i]
		B := nne.Bias[i]

		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level <= minLevel && level < nne.LevelsToComplete(i, false) { //minLevel for Bootstrapping
			if level < minLevel {
				utils.ThrowErr(errors.New(fmt.Sprintf("level below minlevel for bootstrapping: layer %d, level %d", i+1, level)))
			}
			fmt.Println("MinLevel, Bootstrapping...")
			master.StartProto(distributed.REFRESH, Xint, pkQ, minLevel)
			fmt.Println("Level after bootstrapping: ", Xint.Blocks[0][0].Level())
		}

		if i == 0 {
			prepack = false
			Xint = nne.Multiplier.Multiply(Xint, W, prepack)
		} else {
			prepack = true
			Xint = nne.Multiplier.Multiply(Xint, W, prepack)
		}
		//bias
		nne.Adder.AddBias(Xint, B)

		level = Xint.Blocks[0][0].Level()
		if i != len(nne.Weights)-1 {
			if (level < nne.ReLUApprox[i].LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox[i].LevelsOfAct() < minLevel) && level < nne.LevelsToComplete(i, true) {
				if level < minLevel {
					utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
				}
				if level < nne.ReLUApprox[i].LevelsOfAct() {
					fmt.Printf("Level < %d before activation , Bootstrapping...\n", nne.ReLUApprox[i].LevelsOfAct())
				} else if level == minLevel {
					fmt.Println("Min Level , Bootstrapping...")
				} else {
					fmt.Println("Activation would set level below threshold, Pre-emptive Bootstraping...")
					fmt.Println("Curr level: ", level)
					fmt.Println("Drop to: ", minLevel)
					fmt.Println("Diff: ", level-minLevel)
				}
				master.StartProto(distributed.REFRESH, Xint, pkQ, minLevel)
			}
			fmt.Println("Activation")
			//activation
			nne.Activators[i].ActivateBlocks(Xint)
		}
	}
	fmt.Println("Key Switch to querier public key")
	master.StartProto(distributed.CKSWITCH, Xint, pkQ, minLevel)

	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)
	Box := nne.Box
	Box.Decryptor = decQ
	res := cipherUtils.DecInput(Xint, Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)
	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        elapsed,
	}
}
