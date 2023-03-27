//implements neural networks and he neural networks interfaces
package network

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"gonum.org/v1/gonum/mat"
	"time"
)

type HENetworkI interface {
	//Rotations needed for inference. Also updates the box in the network with rotation keys
	GetRotations(params ckks.Parameters, btpParams *bootstrapping.Parameters) []int

	GetBox() cipherUtils.CkksBox
	SetBox(Box cipherUtils.CkksBox)
	//Evaluates batch. Treats each layer l as: Act[l](X * weight[l] + bias[l])
	Eval(X cipherUtils.BlocksOperand) (*cipherUtils.EncInput, time.Duration)
	//Evaluates batch with debug statements. Needs the batch in clear and the network in cleartext.
	//Additionally needs the max L1 norm of the difference allowed for each intermediate result, elementwise.
	//If difference is > L1Thresh, it panics
	EvalDebug(Xenc cipherUtils.BlocksOperand, Xclear *mat.Dense, network NetworkI, L1thresh float64) (*cipherUtils.EncInput, *mat.Dense, time.Duration)
	//Checks if ciphertext at current level, for layer layer, needs bootstrap (true) or not
	//afterMul: if already multiplied by weight
	//forAct: if before activation
	CheckLvlAtLayer(level, minLevel, layer int, forAct, afterMul bool) bool
	//Returns number of levels requested to complete pipeline from current layer
	//afterMul: if already multiplied by weight
	LevelsToComplete(currLayer int, afterMul bool) int
	//True if network is initialized correctly
	IsInit() bool
}

// Network for HE, either in clear or encrypted
// The implemented network should evaluate layers as: Activation(Input * Weight + bias)
type HENetwork struct {
	Weights []cipherUtils.BlocksOperand
	Bias    []cipherUtils.BlocksOperand

	Multiplier *cipherUtils.Multiplier
	Adder      *cipherUtils.Adder
	Activator  *cipherUtils.Activator

	Bootstrapper   cipherUtils.IBootstrapper
	Bootstrappable bool
	MinLevel       int
	BtpCapacity    int
	init           bool
	batchSize      int
	inputInnerRows int
	NumOfLayers    int

	Box cipherUtils.CkksBox
}

// Creates a new network for he inference
// Needs splits of input and weights and loaded network from json
func NewHENetwork(network NetworkI, splits *cipherUtils.Split, encrypted, bootstrappable bool, maxLevel, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) HENetworkI {
	if !network.IsInit() {
		panic("Netowrk in clear is not initialized")
	}
	layers := network.GetNumOfLayers()
	splitInfo, _ := splits.ExctractInfo()
	innerRows := splitInfo.InputRows
	innerCols := splitInfo.InputCols
	inputRowP := splitInfo.InputRowP

	hen := new(HENetwork)

	hen.batchSize = splitInfo.BatchSize
	hen.inputInnerRows = innerRows

	hen.Weights = make([]cipherUtils.BlocksOperand, layers)
	hen.Bias = make([]cipherUtils.BlocksOperand, layers)

	hen.Activator, _ = cipherUtils.NewActivator(network.GetNumOfActivations(), poolsize)
	hen.Multiplier = cipherUtils.NewMultiplier(poolsize)
	hen.Adder = cipherUtils.NewAdder(poolsize)

	hen.Box = Box
	hen.MinLevel = minLevel
	hen.BtpCapacity = btpCapacity
	hen.Bootstrappable = bootstrappable
	hen.NumOfLayers = network.GetNumOfLayers()

	if bootstrappable {
		hen.Bootstrapper = Bootstrapper
	}

	level := maxLevel

	if encrypted {
		fmt.Println("Creating encrypted weights block matrices...")
	} else {
		fmt.Println("Creating encoded weights block matrices...")
	}

	var err error

	splitIdx := 1
	weights, biases := network.GetParamsRescaled()

	for i := 0; i < layers; i++ {

		if network.CheckLvlAtLayer(level, minLevel, i, false, false) {
			if level < minLevel {
				s := fmt.Sprintf("Estimated level below minlevel for layer %d", i+1)
				utils.ThrowErr(errors.New(s))
			}
			if !bootstrappable {
				panic(errors.New("Cannot bootstrap "))
			}
			level = btpCapacity
		}

		split := splits.ExctractInfoAt(splitIdx)
		_, cols, rowP, colP := split[0], split[1], split[2], split[3]
		if encrypted {
			hen.Weights[i], err = cipherUtils.NewEncWeightDiag(weights[i], rowP, colP, innerRows, innerCols, level, Box)
		} else {
			hen.Weights[i], err = cipherUtils.NewPlainWeightDiag(weights[i], rowP, colP, innerRows, innerCols, level, Box)
		}
		innerCols = cols
		utils.ThrowErr(err)
		level-- //mul

		if encrypted {
			hen.Bias[i], err = cipherUtils.NewEncInput(biases[i], inputRowP, colP, level, Box.Params.DefaultScale(), Box)
			utils.ThrowErr(err)
		} else {
			hen.Bias[i], err = cipherUtils.NewPlainInput(biases[i], inputRowP, colP, level, Box.Params.DefaultScale(), Box)
			utils.ThrowErr(err)
		}

		if network.CheckLvlAtLayer(level, minLevel, i, true, true) {
			if level < minLevel {
				s := fmt.Sprintf("Estimated level below minlevel for layer %d", i+1)
				utils.ThrowErr(errors.New(s))
			}
			if !bootstrappable {
				panic(errors.New("Cannot bootstrap "))
			}
			level = btpCapacity
		}

		if i < hen.Activator.NumOfActivations {
			//activation
			var err error
			hen.Activator.AddActivation(network.GetActivations()[i], i, level, Box.Params.DefaultScale(), innerRows, cols, Box)
			utils.ThrowErr(err)
			level -= hen.Activator.LevelsOfAct(i) //activation

			if (level < minLevel && level < network.LevelsToComplete(i+1, true)) || level < 0 {
				if minLevel > 0 {
					panic(errors.New(fmt.Sprintf("Level below minimum level at layer %d\n", i+1)))
				} else {
					panic(errors.New(fmt.Sprintf("Level below 0 at layer %d\n", i+1)))
				}
			}
		}
		splitIdx++
	}
	fmt.Println("Done...")
	hen.init = true
	return hen
}

func (n *HENetwork) IsInit() bool {
	return n.init
}

func (n *HENetwork) GetBox() cipherUtils.CkksBox {
	return n.Box
}

func (n *HENetwork) SetBox(box cipherUtils.CkksBox) {
	n.Box = box
}

//computes how many levels are needed to complete the pipeline in he version
func (n *HENetwork) LevelsToComplete(currLayer int, afterMul bool) int {
	if !n.IsInit() {
		panic(errors.New("Not Inited!"))
	}
	levelsNeeded := 0
	for i := currLayer; i < n.NumOfLayers; i++ {
		levelsNeeded += 1 //mul
		if i < n.Activator.NumOfActivations {
			//last layer with no act
			levelsNeeded += n.Activator.LevelsOfAct(i)
		}
	}
	if afterMul {
		levelsNeeded--
	}
	//fmt.Printf("Levels needed from layer %d to complete: %d\n\n", currLayer+1, levelsNeeded)
	return levelsNeeded
}

//true if he version needs bootstrapping at this layer with level = level
func (n *HENetwork) CheckLvlAtLayer(level, minLevel, layer int, forAct, afterMul bool) bool {
	if !n.IsInit() {
		panic(errors.New("Not Inited!"))
	}
	levelsOfAct := 0
	if layer < n.Activator.NumOfActivations && forAct {
		levelsOfAct = n.Activator.LevelsOfAct(layer)
	}
	return (level < levelsOfAct || level <= minLevel || level-levelsOfAct < minLevel) && level < n.LevelsToComplete(layer, afterMul)
}

//Runs inference
func (n *HENetwork) Eval(X cipherUtils.BlocksOperand) (*cipherUtils.EncInput, time.Duration) {
	if !n.IsInit() {
		panic("Network is not init")
	}
	fmt.Println("Starting inference...")
	start := time.Now()

	level := X.Level()
	res := new(cipherUtils.EncInput)
	for i := 0; i < n.NumOfLayers; i++ {
		layerStart := time.Now()

		fmt.Println("Layer ", i+1)
		if n.CheckLvlAtLayer(level, n.MinLevel, i, false, false) {
			if !n.Bootstrappable {
				panic(errors.New("Needs Bootstrapping but not bootstrappable"))
			}
			n.Bootstrapper.Bootstrap(res, n.Box)
		}
		if i == 0 {
			res = n.Multiplier.Multiply(X, n.Weights[i], false, n.Box)
		} else {
			res = n.Multiplier.Multiply(res, n.Weights[i], true, n.Box)
		}
		n.Adder.AddBias(res, n.Bias[i], n.Box)

		level = res.Level()

		if n.CheckLvlAtLayer(level, n.MinLevel, i, true, true) {
			if !n.Bootstrappable {
				panic(errors.New("Needs Bootstrapping but not bootstrappable"))
			}
			//fmt.Println("Bootstrapping...")
			n.Bootstrapper.Bootstrap(res, n.Box)
			//fmt.Println("Done")
		}

		n.Activator.ActivateBlocks(res, i, n.Box)

		level = res.Level()
		fmt.Println("[*] Layer", i+1, "-Duration ms:", time.Since(layerStart).Milliseconds())
	}
	return res, time.Since(start)
}

//Eval but with debug statements for checking the HE pipeline step by step
func (n *HENetwork) EvalDebug(Xenc cipherUtils.BlocksOperand, Xclear *mat.Dense, network NetworkI, L1thresh float64) (*cipherUtils.EncInput, *mat.Dense, time.Duration) {
	if !n.IsInit() {
		panic("Network is not init")
	}
	fmt.Println("Starting inference DEBUG...")
	start := time.Now()

	level := Xenc.Level()
	res := new(cipherUtils.EncInput)

	resClear := Xclear

	w, b := network.GetParamsRescaled()
	activations := network.GetActivations()

	for i := 0; i < n.NumOfLayers; i++ {
		layerStart := time.Now()
		fmt.Println("Layer ", i+1)
		if n.CheckLvlAtLayer(level, n.MinLevel, i, false, false) {
			if !n.Bootstrappable {
				panic(errors.New("Needs Bootstrapping but not bootstrappable"))
			}
			fmt.Println("Bootstrapping...")
			tBtp := time.Now()
			n.Bootstrapper.Bootstrap(res, n.Box)
			fmt.Println("Done. Time: ", time.Since(tBtp).Milliseconds())
		}
		tMul := time.Now()
		if i == 0 {
			res = n.Multiplier.Multiply(Xenc, n.Weights[i], false, n.Box)
		} else {
			res = n.Multiplier.Multiply(res, n.Weights[i], true, n.Box)
		}
		eMul := time.Since(tMul).Milliseconds()
		var tmp mat.Dense
		tmp.Mul(resClear, w[i])
		tmpBlocks, err := plainUtils.PartitionMatrix(&tmp, res.RowP, res.ColP)
		utils.ThrowErr(err)
		cipherUtils.PrintDebugBlocks(res, tmpBlocks, L1thresh, n.Box)
		fmt.Println("^^^^^^^^^^^^^^^^^^^^^^^^")
		fmt.Printf("Multiplication layer %d. Time: %d\n", i+1, eMul)

		tBias := time.Now()
		n.Adder.AddBias(res, n.Bias[i], n.Box)
		eBias := time.Since(tBias).Milliseconds()

		var tmp2 mat.Dense
		tmp2.Add(&tmp, b[i])
		tmpBlocks, err = plainUtils.PartitionMatrix(&tmp2, res.RowP, res.ColP)
		cipherUtils.PrintDebugBlocks(res, tmpBlocks, L1thresh, n.Box)
		fmt.Println("^^^^^^^^^^^^^^")
		fmt.Printf("Bias layer %d. Time %d\n", i+1, eBias)

		level = res.Level()

		if n.CheckLvlAtLayer(level, n.MinLevel, i, true, true) {
			if !n.Bootstrappable {
				panic(errors.New("Needs Bootstrapping but not bootstrappable"))
			}
			fmt.Println("Bootstrapping...")
			tBtp := time.Now()
			n.Bootstrapper.Bootstrap(res, n.Box)
			fmt.Println("Done. Time: ", time.Since(tBtp).Milliseconds())
		}
		tAct := time.Now()
		n.Activator.ActivateBlocks(res, i, n.Box)
		eAct := time.Since(tAct).Milliseconds()
		if i < network.GetNumOfActivations() {
			activations[i].ActivatePlain(&tmp2)
			tmpBlocks, err = plainUtils.PartitionMatrix(&tmp2, res.RowP, res.ColP)
			cipherUtils.PrintDebugBlocks(res, tmpBlocks, L1thresh, n.Box)
			fmt.Println("^^^^^^^^^^^^^^^^^^^^")
			fmt.Printf("Activation layer %d. Time: %d\n", i+1, eAct)
		}
		*resClear = tmp2
		level = res.Level()
		fmt.Println("[*] Layer", i+1, "-Duration ms:", time.Since(layerStart).Milliseconds())
	}
	return res, resClear, time.Since(start)
}

//Generate the rotations needed to evaluate the network and updates the network box with the rotation keys
func (n *HENetwork) GetRotations(params ckks.Parameters, btpParams *bootstrapping.Parameters) []int {
	rs := cipherUtils.NewRotationsSet()
	for i, w := range n.Weights {
		rs.Add(w.GetRotations(params))
		if (i + 1) < len(n.Weights) {
			currColp, _ := n.Weights[i].GetPartitions()
			_, currInCols := n.Weights[i].GetInnerDims()
			_, currCols := n.Weights[i].GetRealDims()
			_, nextRowP := n.Weights[i+1].GetPartitions()
			if currColp != nextRowP {
				rs.Add(cipherUtils.GenRotationsForRepackCols(n.inputInnerRows, currCols, currInCols, nextRowP))
			}
		}

	}
	rots := rs.Rotations()
	n.Box = cipherUtils.BoxWithRotations(n.Box, rots, n.Bootstrappable, btpParams)

	return rots
}
