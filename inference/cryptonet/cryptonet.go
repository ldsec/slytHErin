package cryptonet

import "C"
import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	cU "github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	utils "github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"os"
	"time"
)

type cryptonet struct {
	Conv1 utils.Layer `json:"conv1"`
	Pool1 utils.Layer `json:"pool1"`
	Pool2 utils.Layer `json:"pool2"`

	ReLUApprox *utils.MinMaxPolyApprox //this will store the coefficients of the poly approximating ReLU
}

//holds encoded model for encrypted data inference
type cryptonetEcd struct {
	Weights    []*cU.PlainWeightDiag
	Bias       []*cU.PlainInput
	Activators []*cU.Activator
	Multiplier *cU.Multiplier
	Adder      *cU.Adder
	Box        cU.CkksBox
}

/***************************
HELPERS
 ***************************/
func Loadcryptonet(path string) *cryptonet {
	// loads json file with weights
	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res cryptonet
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}

/****************
cryptonet METHODS
 ***************/

func (sn *cryptonet) Init() {
	deg := 3
	sn.ReLUApprox = utils.InitReLU(deg)
}

//returns list of (weights, biases) as arrays of *mat.Dense
func (sn *cryptonet) BuildParams(batchSize int) ([]*mat.Dense, []*mat.Dense) {
	conv1M := utils.BuildKernelMatrix(sn.Conv1.Weight)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := utils.BuildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)
	pool1M := utils.BuildKernelMatrix(sn.Pool1.Weight)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := utils.BuildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)
	pool2M := utils.BuildKernelMatrix(sn.Pool2.Weight)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := utils.BuildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)
	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	return weightMatrices, biasMatrices
}

//Compress the 2 linear layers back to back
func (sn *cryptonet) CompressLayers(weights, biases []*mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	//plaintext weights and bias
	conv1M := weights[0]
	pool1M := weights[1]
	bias1M := biases[0]
	bias2M := biases[1]

	var compressed mat.Dense
	compressed.Mul(conv1M, pool1M)
	var biasCompressed mat.Dense
	biasCompressed.Mul(bias1M, pool1M)
	biasCompressed.Add(&biasCompressed, bias2M)

	return []*mat.Dense{&compressed, weights[2]}, []*mat.Dense{&biasCompressed, biases[2]}
}

//Rescale the Parameters by 1/interval
func (sn *cryptonet) RescaleForActivation(weights, biases []*mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	wRescaled := make([]*mat.Dense, len(weights))
	bRescaled := make([]*mat.Dense, len(biases))

	for i := range weights {
		wRescaled[i] = new(mat.Dense)
		bRescaled[i] = new(mat.Dense)
		if i != len(weights)-1 {
			wRescaled[i].Scale(float64(1.0/sn.ReLUApprox.Interval), weights[i])
			bRescaled[i].Scale(float64(1.0/sn.ReLUApprox.Interval), biases[i])
		} else {
			wRescaled[i] = weights[i]
			bRescaled[i] = biases[i]
		}
		fmt.Printf("Layer %d:\n", i+1)
		fmt.Printf("Dense Weight Dims (R,C): %d,%d\n", plainUtils.NumRows(wRescaled[i]), plainUtils.NumCols(wRescaled[i]))
		fmt.Printf("Dense Bias Dims (R,C): %d,%d\n", plainUtils.NumRows(bRescaled[i]), plainUtils.NumCols(bRescaled[i]))
	}
	return wRescaled, bRescaled
}

func (sn *cryptonet) Encodecryptonet(weights, biases []*mat.Dense, splits []cU.BlockSplits, Box cU.CkksBox, poolsize int) *cryptonetEcd {
	sne := new(cryptonetEcd)

	//compress layers 1 and 2 which are linear and back to back, then rescale for the activation
	weights, biases = sn.RescaleForActivation(weights, biases)

	sne.Weights = make([]*cU.PlainWeightDiag, len(weights))
	sne.Bias = make([]*cU.PlainInput, len(biases))
	sne.Activators = make([]*cU.Activator, 2)

	level := Box.Params.MaxLevel()
	scale := Box.Params.DefaultScale()
	var err error

	leftInnerDim := splits[0].InnerRows
	inputRowP := splits[0].RowP

	iAct := 0

	for i, split := range splits[1:] {
		sne.Weights[i], err = cU.NewPlainWeightDiag(weights[i], split.RowP, split.ColP, leftInnerDim, level, Box)
		utils.ThrowErr(err)

		level-- //rescale

		sne.Bias[i], err = cU.NewPlainInput(biases[i], inputRowP, split.ColP, level, scale, Box)
		utils.ThrowErr(err)

		if iAct < 2 {
			sne.Activators[iAct], err = cU.NewActivator(sn.ReLUApprox, level, scale, leftInnerDim, split.InnerCols, Box, poolsize)
			utils.ThrowErr(err)
			iAct++
			level -= sn.ReLUApprox.LevelsOfAct()
		}
	}
	sne.Adder = cU.NewAdder(Box, poolsize)
	sne.Multiplier = cU.NewMultiplier(Box, poolsize)
	sne.Box = Box

	return sne
}

func (sne *cryptonetEcd) EvalBatchEncrypted(Xenc *cU.EncInput, Y []int, labels int) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	iAct := 0
	for i := range sne.Weights {
		Xenc = sne.Multiplier.Multiply(Xenc, sne.Weights[i])
		sne.Multiplier.RemoveImagFromBlocks(Xenc)

		sne.Adder.AddBias(Xenc, sne.Bias[i])

		if iAct < 2 {
			sne.Activators[iAct].ActivateBlocks(Xenc)
			iAct++
		}
	}
	end := time.Since(start)
	fmt.Println("Done ", end)
	res := cU.DecInput(Xenc, sne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)

	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        end,
	}
}

func (sne *cryptonetEcd) EvalBatchEncrypted_Debug(Xenc *cU.EncInput, Xclear *mat.Dense, weights, biases []*mat.Dense, activation *utils.MinMaxPolyApprox, Y []int, labels int) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	iAct := 0

	for i := range sne.Weights {

		Xenc = sne.Multiplier.Multiply(Xenc, sne.Weights[i])
		sne.Multiplier.RemoveImagFromBlocks(Xenc)

		var tmp mat.Dense
		tmp.Mul(Xclear, weights[i])
		tmpRescaled := plainUtils.MulByConst(&tmp, 1.0/activation.Interval)
		tmpB, _ := plainUtils.PartitionMatrix(tmpRescaled, Xenc.RowP, Xenc.ColP)
		cU.PrintDebugBlocks(Xenc, tmpB, 0.1, sne.Box)

		sne.Adder.AddBias(Xenc, sne.Bias[i])

		var tmp2 mat.Dense
		tmp2.Add(&tmp, biases[i])
		tmpRescaled = plainUtils.MulByConst(&tmp2, 1.0/activation.Interval)
		tmpB, _ = plainUtils.PartitionMatrix(tmpRescaled, Xenc.RowP, Xenc.ColP)
		cU.PrintDebugBlocks(Xenc, tmpB, 0.1, sne.Box)

		if iAct < 2 {
			sne.Activators[iAct].ActivateBlocks(Xenc)
			utils.ActivatePlain(&tmp2, activation)
			tmpB, _ = plainUtils.PartitionMatrix(&tmp2, Xenc.RowP, Xenc.ColP)
			cU.PrintDebugBlocks(Xenc, tmpB, 1, sne.Box)
		}
		iAct++
		*Xclear = tmp2
	}
	end := time.Since(start)
	fmt.Println("Done ", end)
	res := cU.DecInput(Xenc, sne.Box)
	corrects, accuracy, predictions := utils.Predict(Y, labels, res)

	return utils.Stats{
		Predictions: predictions,
		Corrects:    corrects,
		Accuracy:    accuracy,
		Time:        end,
	}
}
