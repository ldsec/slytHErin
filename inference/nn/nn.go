package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"os"
	"time"
)

/*
	Stores NN model in clear, as in json format
*/
type NN struct {
	Conv       utils.Layer            `json:"conv"`
	Dense      []utils.Layer          `json:"dense"`
	Layers     int                    `json:"layers"`
	ReLUApprox *utils.ChebyPolyApprox //this will store the coefficients of the poly approximating ReLU

	RowsOutConv, ColsOutConv, ChansOutConv, DimOutDense int //dimentions
}

/*
	Wrapper for Encrypted layers in Block Matrix form
*/
type NNEnc struct {
	Weights    []*cipherUtils.EncWeightDiag
	Bias       []*cipherUtils.EncInput
	Activators []*cipherUtils.Activator
	ReLUApprox *utils.ChebyPolyApprox //this will store the coefficients of the poly approximating ReLU

	Multiplier *cipherUtils.Multiplier
	Adder      *cipherUtils.Adder

	Box    cipherUtils.CkksBox
	Layers int
}

//Approximation parameters for Chebychev approximated activations. Depends on the number of layers
type ApproxParams struct {
	a, b float64
	deg  int
}

var NN20Params = ApproxParams{a: -35, b: 35, deg: 63}
var NN20Params_CentralizedBtp = ApproxParams{a: -35, b: 35, deg: 3} //deg needs to be < residual capacity
var NN50Params = ApproxParams{a: -55, b: 55, deg: 63}

//computes how many levels are needed to complete the pipeline
func (nne NNEnc) LevelsToComplete(currLayer int, afterMul bool) int {
	levelsNeeded := 0
	for i := currLayer; i < nne.Layers+1; i++ {
		levelsNeeded += 1 //mul
		if i != nne.Layers {
			//last layer with no act
			levelsNeeded += nne.ReLUApprox.LevelsOfAct()
		}
	}
	if afterMul {
		levelsNeeded--
	}
	//fmt.Printf("Levels needed from layer %d to complete: %d\n\n", currLayer+1, levelsNeeded)
	return levelsNeeded
}

// loads json file with weights
func LoadNN(path string) *NN {

	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res NN
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}

func (nn *NN) Init(layers int, distributedBtp bool) {
	//init dimensional values (not really used, just for reference)
	nn.Layers = layers
	//nn.RowsOutConv = 21
	//nn.ColsOutConv = 20
	//nn.ChansOutConv = 2 //tot is 21*20*2 = 840 per sample after conv
	//nn.DimOutDense = 92 //per sample, all dense are the same but last one which is 92x10
	if layers == 20 {
		if distributedBtp {
			//distributed
			nn.ReLUApprox = utils.InitActivationCheby("soft relu", NN20Params.a, NN20Params.b, NN20Params.deg)
		} else {
			nn.ReLUApprox = utils.InitActivationCheby("soft relu", NN20Params_CentralizedBtp.a, NN20Params_CentralizedBtp.b, NN20Params_CentralizedBtp.deg)
		}
	} else if layers == 50 {
		if distributedBtp {
			//distributed
			nn.ReLUApprox = utils.InitActivationCheby("soft relu", NN50Params.a, NN50Params.b, NN50Params.deg)
		} else {
			nn.ReLUApprox = utils.InitActivationCheby("soft relu", NN20Params_CentralizedBtp.a, NN20Params_CentralizedBtp.b, NN20Params_CentralizedBtp.deg)
		}
	}
}

func (nn *NN) BuildParams(batchSize int) ([]*mat.Dense, []*mat.Dense) {
	layers := nn.Layers

	denseMatrices := make([]*mat.Dense, layers+1)
	denseBiasMatrices := make([]*mat.Dense, layers+1)

	convM := utils.BuildKernelMatrix(nn.Conv.Weight)
	biasConvM := utils.BuildBiasMatrix(nn.Conv.Bias, plainUtils.NumCols(convM), batchSize)
	denseMatrices[0] = convM
	denseBiasMatrices[0] = biasConvM

	for i := 0; i < layers; i++ {
		denseMatrices[i+1] = utils.BuildKernelMatrix(nn.Dense[i].Weight)
		denseBiasMatrices[i+1] = utils.BuildBiasMatrix(nn.Dense[i].Bias, plainUtils.NumCols(denseMatrices[i+1]), batchSize)
	}
	return denseMatrices, denseBiasMatrices
}

func (nn *NN) RescaleWeightsForActivation(weights, biases []*mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	scaledW := make([]*mat.Dense, len(weights))
	scaledB := make([]*mat.Dense, len(biases))
	for i := range weights {
		//change to cheby base
		a := nn.ReLUApprox.A
		b := nn.ReLUApprox.B
		mulC := 2 / (b - a)
		addC := (-a - b) / (b - a)
		if i == nn.Layers {
			//skip base switch for activation, since there is none in last layer
			mulC = 1.0
			addC = 0.0
		}
		scaledW[i] = plainUtils.MulByConst(weights[i], mulC)
		scaledB[i] = plainUtils.AddConst(plainUtils.MulByConst(biases[i], mulC), addC)
	}
	return scaledW, scaledB
}

//Forms an encrypted NN from the plaintext representation. Set minlevel -1 and btpCapacity whatever if centralized bootstrapping
func (nn *NN) EncryptNN(weights, biases []*mat.Dense, splits []cipherUtils.BlockSplits, btpCapacity int, minLevel int, Box cipherUtils.CkksBox, poolsize int) (*NNEnc, error) {
	layers := nn.Layers
	splitInfo := cipherUtils.ExctractInfo(splits)
	innerRows := splitInfo.InputRows
	inputRowP := splitInfo.InputRowP
	nne := new(NNEnc)
	nne.Weights = make([]*cipherUtils.EncWeightDiag, layers+1)
	nne.Bias = make([]*cipherUtils.EncInput, layers+1)
	nne.Activators = make([]*cipherUtils.Activator, layers)
	nne.Layers = nn.Layers
	nne.ReLUApprox = nn.ReLUApprox
	nne.Box = Box

	maxLevel := Box.Params.MaxLevel()
	level := maxLevel

	fmt.Println("Creating weights encrypted block matrices...")
	var err error

	splitIdx := 1
	for i := 0; i < layers+1; i++ {

		if ((level <= minLevel && minLevel != -1) || level == 0) && level < nne.LevelsToComplete(i, false) {
			//bootstrap
			if minLevel != -1 {
				//distributed
				if level < minLevel {
					s := fmt.Sprintf("Estimated level below minlevel for layer %d", i+1)
					utils.ThrowErr(errors.New(s))
				}
				level = maxLevel
			} else {
				//centralized
				level = btpCapacity
			}
		}

		split := splits[splitIdx]
		nne.Weights[i], err = cipherUtils.NewEncWeightDiag(weights[i], split.RowP, split.ColP, innerRows, level, Box)
		level-- //mul

		if (level < minLevel && level < nne.LevelsToComplete(i, true)) || level < 0 {
			if minLevel > 0 {
				panic(errors.New(fmt.Sprintf("Level below minimum level at layer %d\n", i+1)))
			} else {
				panic(errors.New(fmt.Sprintf("Level below 0 at layer %d\n", i+1)))
			}
		}

		nne.Bias[i], err = cipherUtils.NewEncInput(biases[i], inputRowP, split.ColP, level, Box)
		utils.ThrowErr(err)

		if (level < nne.ReLUApprox.LevelsOfAct() || (minLevel != -1 && (level < nne.ReLUApprox.LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox.LevelsOfAct() < minLevel))) && level < nne.LevelsToComplete(i, true) {
			if minLevel != -1 {
				//distributed
				if level < minLevel {
					s := fmt.Sprintf("Estimated level below minlevel for layer %d", i+1)
					utils.ThrowErr(errors.New(s))
				}
				level = maxLevel
			} else {
				//centralized
				level = btpCapacity
			}
		}

		if i != layers {
			//activation
			var err error
			nne.Activators[i], err = cipherUtils.NewActivator(nn.ReLUApprox, level, Box.Params.DefaultScale(), innerRows, split.InnerCols, Box, poolsize)
			utils.ThrowErr(err)
			level -= nne.ReLUApprox.LevelsOfAct() //activation

			if (level < minLevel && level < nne.LevelsToComplete(i+1, true)) || level < 0 {
				if minLevel > 0 {
					panic(errors.New(fmt.Sprintf("Level below minimum level at layer %d\n", i+1)))
				} else {
					panic(errors.New(fmt.Sprintf("Level below 0 at layer %d\n", i+1)))
				}
			}
		}

		splitIdx++
	}
	nne.Multiplier = cipherUtils.NewMultiplier(Box, poolsize)
	nne.Adder = cipherUtils.NewAdder(Box, poolsize)
	fmt.Println("Done...")
	return nne, nil
}

func (nne *NNEnc) EvalBatchEncrypted_Debug(Xenc *cipherUtils.EncInput, Y []int, Xclear *mat.Dense, weights, biases []*mat.Dense, activation func(float64) float64, labels int, Btp *cipherUtils.Bootstrapper) utils.Stats {
	Xint := Xenc
	XintPlain := Xclear
	Box := nne.Box

	now := time.Now()
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
		Xint = nne.Multiplier.Multiply(Xint, W)

		a := nne.ReLUApprox.A
		b := nne.ReLUApprox.B
		mulC := 2 / (b - a)
		addC := (-a - b) / (b - a)
		if i == nne.Layers {
			//skip base switch for activation, since there is none in last layer
			mulC = 1.0
			addC = 0.0
		}
		var tmp mat.Dense
		tmp.Mul(XintPlain, weights[i])
		tmpRescaled := plainUtils.MulByConst(&tmp, mulC)
		tmpBlocks, err := plainUtils.PartitionMatrix(tmpRescaled, Xint.RowP, Xint.ColP)
		utils.ThrowErr(err)
		fmt.Printf("Mul ")
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 0.001, Box)

		//bias
		nne.Adder.AddBias(Xint, B)
		utils.ThrowErr(err)

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpRescaled = plainUtils.MulByConst(&tmp2, mulC)
		tmpRescaled = plainUtils.AddConst(tmpRescaled, addC)
		tmpBlocks, err = plainUtils.PartitionMatrix(tmpRescaled, Xint.RowP, Xint.ColP)

		//activation
		if i != len(nne.Weights)-1 {
			level = Xint.Blocks[0][0].Level()
			if level < nne.ReLUApprox.LevelsOfAct() {
				fmt.Println("Bootstrapping for Activation")
				fmt.Println("pre boot")
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
				Btp.Bootstrap(Xint)
				fmt.Println("after boot")
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			}
			fmt.Printf("Activation ")

			nne.Activators[i].ActivateBlocks(Xint)

			XintPlain = plainUtils.ApplyFuncDense(activation, &tmp2)
			XintPlainBlocks, _ := plainUtils.PartitionMatrix(XintPlain, Xint.RowP, Xint.ColP)
			cipherUtils.PrintDebugBlocks(Xint, XintPlainBlocks, 0.001, Box)
		}
	}
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	res := cipherUtils.DecInput(Xenc, Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)
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
		Xint = nne.Multiplier.Multiply(Xint, W)

		//bias
		nne.Adder.AddBias(Xint, B)

		//activation
		if i != len(nne.Weights)-1 {
			level = Xint.Blocks[0][0].Level()
			if level < nne.ReLUApprox.LevelsOfAct() {
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

	res := cipherUtils.DecInput(Xenc, nne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)
	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        elapsed,
	}
}

func (nne *NNEnc) EvalBatchEncrypted_Distributed_Debug(Xenc *cipherUtils.EncInput, Y []int, Xclear *mat.Dense, weights, biases []*mat.Dense, activation func(float64) float64, labels int, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int,
	master *distributed.LocalMaster) utils.Stats {

	Xint := Xenc
	XintPlain := Xclear

	now := time.Now()

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
		Xint = nne.Multiplier.Multiply(Xint, W)

		a := nne.ReLUApprox.A
		b := nne.ReLUApprox.B
		mulC := 2 / (b - a)
		addC := (-a - b) / (b - a)
		if i == nne.Layers {
			//skip base switch for activation, since there is none in last layer
			mulC = 1.0
			addC = 0.0
		}
		var tmp mat.Dense
		tmp.Mul(XintPlain, weights[i])
		tmpRescaled := plainUtils.MulByConst(&tmp, mulC)
		tmpBlocks, err := plainUtils.PartitionMatrix(tmpRescaled, Xint.RowP, Xint.ColP)
		utils.ThrowErr(err)
		fmt.Printf("Mul ")
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 0.001, nne.Box)

		//bias
		nne.Adder.AddBias(Xint, B)

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpRescaled = plainUtils.MulByConst(&tmp2, mulC)
		tmpRescaled = plainUtils.AddConst(tmpRescaled, addC)
		tmpBlocks, err = plainUtils.PartitionMatrix(tmpRescaled, Xint.RowP, Xint.ColP)
		cipherUtils.PrintDebugBlocks(Xint, tmpBlocks, 0.001, nne.Box)

		if i != len(nne.Weights)-1 {
			if (level < nne.ReLUApprox.LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox.LevelsOfAct() < minLevel) && level < nne.LevelsToComplete(i, true) {
				if level < minLevel {
					utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
				}
				if level < nne.ReLUApprox.LevelsOfAct() {
					fmt.Printf("Level < %d before activation , Bootstrapping...\n", nne.ReLUApprox.LevelsOfAct())
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
			XintPlain = plainUtils.ApplyFuncDense(activation, &tmp2)
			XintPlainBlocks, _ := plainUtils.PartitionMatrix(XintPlain, Xint.RowP, Xint.ColP)
			cipherUtils.PrintDebugBlocks(Xint, XintPlainBlocks, 0.001, nne.Box)
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

func (nne *NNEnc) EvalBatchEncrypted_Distributed(Xenc *cipherUtils.EncInput, Y []int, labels int, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int,
	master *distributed.LocalMaster) utils.Stats {

	Xint := Xenc

	now := time.Now()

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
		Xint = nne.Multiplier.Multiply(Xint, W)

		//bias
		nne.Adder.AddBias(Xint, B)

		if i != len(nne.Weights)-1 {
			if (level < nne.ReLUApprox.LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox.LevelsOfAct() < minLevel) && level < nne.LevelsToComplete(i, true) {
				if level < minLevel {
					utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
				}
				if level < nne.ReLUApprox.LevelsOfAct() {
					fmt.Printf("Level < %d before activation , Bootstrapping...\n", nne.ReLUApprox.LevelsOfAct())
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
