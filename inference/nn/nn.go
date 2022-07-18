package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math"
	"os"
)

/*
	Stores NN model in clear, as in json format
*/
type NN struct {
	Conv       utils.Layer              `json:"conv"`
	Dense      []utils.Layer            `json:"dense"`
	Layers     int                      `json:"layers"`
	ReLUApprox []*utils.ChebyPolyApprox //this will store the coefficients of the poly approximating ReLU

	RowsOutConv, ColsOutConv, ChansOutConv, DimOutDense int //dimentions
}

type NNEnc struct {
	//Wrapper for Encrypted layers in Block Matrix form

	Weights    []*cipherUtils.EncWeightDiag
	Bias       []*cipherUtils.EncInput
	Activators []*cipherUtils.Activator
	ReLUApprox []*utils.ChebyPolyApprox

	Multiplier *cipherUtils.Multiplier
	Adder      *cipherUtils.Adder

	Box    cipherUtils.CkksBox
	Layers int
}

type NNEcd struct {
	//Wrapper for Plaintext layers in Block Matrix form

	Weights    []*cipherUtils.PlainWeightDiag
	Bias       []*cipherUtils.PlainInput
	Activators []*cipherUtils.Activator
	ReLUApprox []*utils.ChebyPolyApprox //this will store the coefficients of the poly approximating ReLU

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
func (nne *NNEnc) LevelsToComplete(currLayer int, afterMul bool) int {
	levelsNeeded := 0
	for i := currLayer; i < nne.Layers+1; i++ {
		levelsNeeded += 1 //mul
		if i != nne.Layers {
			//last layer with no act
			levelsNeeded += nne.ReLUApprox[i].LevelsOfAct()
		}
	}
	if afterMul {
		levelsNeeded--
	}
	//fmt.Printf("Levels needed from layer %d to complete: %d\n\n", currLayer+1, levelsNeeded)
	return levelsNeeded
}

//computes how many levels are needed to complete the pipeline
func (nne *NNEcd) LevelsToComplete(currLayer int, afterMul bool) int {
	levelsNeeded := 0
	for i := currLayer; i < nne.Layers+1; i++ {
		levelsNeeded += 1 //mul
		if i != nne.Layers {
			//last layer with no act
			levelsNeeded += nne.ReLUApprox[i].LevelsOfAct()
		}
	}
	if afterMul {
		levelsNeeded--
	}
	//fmt.Printf("Levels needed from layer %d to complete: %d\n\n", currLayer+1, levelsNeeded)
	return levelsNeeded
}

// loads json file with weights
func LoadNN(layers int, HEtrain bool) *NN {

	var path string
	var suffix string
	if HEtrain {
		suffix = "_poly"
	} else {
		suffix = ""
	}
	path = fmt.Sprintf("nn%d%s_packed.json", layers, suffix)

	jsonFile, err := os.Open(path)
	if err != nil {
		utils.ThrowErr(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res NN
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}

//decides the degree of approximation for each interval
func SetDegOfInterval(intervals utils.ApproxIntervals) utils.ApproxIntervals {
	intervalsNew := make([]utils.ApproxInterval, len(intervals.Intervals))
	margin := 2.0
	for i, interval := range intervals.Intervals {
		interval.A = math.Floor(interval.A) - margin
		interval.B = math.Floor(interval.B) + margin
		diff := interval.B - interval.A
		if diff <= 2 {
			interval.Deg = 3
		} else if diff <= 4 {
			interval.Deg = 7
		} else if diff <= 8 {
			interval.Deg = 15
		} else if diff <= 16 {
			interval.Deg = 31
		} else {
			interval.Deg = 63
			if i == 1 { //layer 2
				interval.Deg = 31
			}
		}
		fmt.Printf("Layer %d Approx: A = %f, B=%f --> deg = %d\n", i+1, interval.A, interval.B, interval.Deg)
		intervalsNew[i] = interval
	}
	return utils.ApproxIntervals{intervalsNew}
}

//init activations. Set HEtrain if model is the one for train under HE
func (nn *NN) Init(layers int, distributedBtp bool, HEtrain bool) {

	nn.Layers = layers
	nn.ReLUApprox = make([]*utils.ChebyPolyApprox, layers)
	var suffix string
	var act string
	if HEtrain {
		suffix = "_poly"
		act = "silu"
	} else {
		suffix = ""
		act = "soft relu"
	}
	jsonFile, err := os.Open(fmt.Sprintf("nn%d%s_intervals.json", layers, suffix))
	if err != nil { //|| !distributedBtp {
		if err != nil {
			fmt.Println("Couldn't open intervals file")
		}
		//default approximation
		if layers == 20 {
			if distributedBtp {
				//distributed
				nn.ReLUApprox[0] = utils.InitActivationCheby(act, NN20Params.a, NN20Params.b, NN20Params.deg)
			} else {
				nn.ReLUApprox[0] = utils.InitActivationCheby(act, NN20Params_CentralizedBtp.a, NN20Params_CentralizedBtp.b, NN20Params_CentralizedBtp.deg)
			}
		} else if layers == 50 {
			if distributedBtp {
				//distributed
				nn.ReLUApprox[0] = utils.InitActivationCheby(act, NN50Params.a, NN50Params.b, NN50Params.deg)
			} else {
				nn.ReLUApprox[0] = utils.InitActivationCheby(act, NN20Params_CentralizedBtp.a, NN20Params_CentralizedBtp.b, NN20Params_CentralizedBtp.deg)
			}
		}
	} else {
		defer jsonFile.Close()
		byteValue, _ := ioutil.ReadAll(jsonFile)
		var intervals utils.ApproxIntervals
		json.Unmarshal([]byte(byteValue), &intervals)
		intervals = SetDegOfInterval(intervals)
		for i := range intervals.Intervals {
			interval := intervals.Intervals[i]
			nn.ReLUApprox[i] = utils.InitActivationCheby(act, interval.A, interval.B, interval.Deg)
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
	for i := range weights[:len(weights)-1] {
		//skip last as no activation
		scaledW[i], scaledB[i] = nn.ReLUApprox[i].Rescale(weights[i], biases[i])
	}
	scaledW[len(weights)-1], scaledB[len(weights)-1] = weights[len(weights)-1], biases[len(weights)-1]
	return scaledW, scaledB
}

//Forms an encrypted NN from the plaintext representation. Set minlevel -1 and btpCapacity whatever if centralized bootstrapping
func (nn *NN) EncryptNN(weights, biases []*mat.Dense, splits []cipherUtils.BlockSplits, btpCapacity int, minLevel int, Box cipherUtils.CkksBox, poolsize int) (*NNEnc, error) {

	layers := nn.Layers
	splitInfo, _ := cipherUtils.ExctractInfo(splits)
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
		levelsOfAct := 0
		if i != layers {
			levelsOfAct = nne.ReLUApprox[i].LevelsOfAct()
		}

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

		nne.Bias[i], err = cipherUtils.NewEncInput(biases[i], inputRowP, split.ColP, level, Box.Params.DefaultScale(), Box)
		utils.ThrowErr(err)

		if (level < levelsOfAct || (minLevel != -1 && (level < levelsOfAct || level <= minLevel || level-levelsOfAct < minLevel))) && level < nne.LevelsToComplete(i, true) {
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
			nne.Activators[i], err = cipherUtils.NewActivator(nn.ReLUApprox[i], level, Box.Params.DefaultScale(), innerRows, split.InnerCols, Box, poolsize)
			utils.ThrowErr(err)
			level -= nne.ReLUApprox[i].LevelsOfAct() //activation

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

//Forms an encoded plaintext NN from the plaintext representation. Set minlevel -1 and btpCapacity whatever if centralized bootstrapping
func (nn *NN) EncodeNN(weights, biases []*mat.Dense, splits []cipherUtils.BlockSplits, btpCapacity int, minLevel int, Box cipherUtils.CkksBox, poolsize int) (*NNEcd, error) {

	layers := nn.Layers
	splitInfo, _ := cipherUtils.ExctractInfo(splits)
	innerRows := splitInfo.InputRows
	inputRowP := splitInfo.InputRowP
	nne := new(NNEcd)
	nne.Weights = make([]*cipherUtils.PlainWeightDiag, layers+1)
	nne.Bias = make([]*cipherUtils.PlainInput, layers+1)
	nne.Activators = make([]*cipherUtils.Activator, layers)
	nne.Layers = nn.Layers
	nne.ReLUApprox = nn.ReLUApprox
	nne.Box = Box

	maxLevel := Box.Params.MaxLevel()
	level := maxLevel

	fmt.Println("Creating weights encoded block matrices...")
	var err error

	splitIdx := 1
	for i := 0; i < layers+1; i++ {
		levelsOfAct := 0
		if i != layers {
			levelsOfAct = nne.ReLUApprox[i].LevelsOfAct()
		}

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
		nne.Weights[i], err = cipherUtils.NewPlainWeightDiag(weights[i], split.RowP, split.ColP, innerRows, level, Box)
		level-- //mul

		if (level < minLevel && level < nne.LevelsToComplete(i, true)) || level < 0 {
			if minLevel > 0 {
				panic(errors.New(fmt.Sprintf("Level below minimum level at layer %d\n", i+1)))
			} else {
				panic(errors.New(fmt.Sprintf("Level below 0 at layer %d\n", i+1)))
			}
		}

		nne.Bias[i], err = cipherUtils.NewPlainInput(biases[i], inputRowP, split.ColP, level, Box.Params.DefaultScale(), Box)
		utils.ThrowErr(err)

		if (level < levelsOfAct || (minLevel != -1 && (level < levelsOfAct || level <= minLevel || level-levelsOfAct < minLevel))) && level < nne.LevelsToComplete(i, true) {
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
			nne.Activators[i], err = cipherUtils.NewActivator(nn.ReLUApprox[i], level, Box.Params.DefaultScale(), innerRows, split.InnerCols, Box, poolsize)
			utils.ThrowErr(err)
			level -= nne.ReLUApprox[i].LevelsOfAct() //activation

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
