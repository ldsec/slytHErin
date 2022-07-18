package cryptonet

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
)

type Cryptonet struct {
	Conv1 utils.Layer `json:"conv1"`
	Pool1 utils.Layer `json:"pool1"`
	Pool2 utils.Layer `json:"pool2"`

	ReLUApprox *utils.ChebyPolyApprox
}

//holds encoded model for encrypted data inference
type CryptonetEcd struct {
	Weights    []*cU.PlainWeightDiag
	Bias       []*cU.PlainInput
	Activators []*cU.Activator
	Multiplier *cU.Multiplier
	Adder      *cU.Adder
	Box        cU.CkksBox
}

//holds encrypted model for data inference (scenario model in clear, data encrypted)
type CryptonetEnc struct {
	Weights    []*cU.EncWeightDiag
	Bias       []*cU.EncInput
	Activators []*cU.Activator
	Multiplier *cU.Multiplier
	Adder      *cU.Adder
	Box        cU.CkksBox
}

func LoadCryptonet(path string) *Cryptonet {
	// loads json file with weights
	jsonFile, err := os.Open(path)
	utils.ThrowErr(err)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res Cryptonet
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}

/****************
Cryptonet METHODS
 ***************/

func (cn *Cryptonet) Init() {
	deg := 3
	cn.ReLUApprox = utils.InitReLU(deg)
}

//returns list of (weights, biases) as arrays of *mat.Dense
func (cn *Cryptonet) BuildParams(batchSize int) ([]*mat.Dense, []*mat.Dense) {
	conv1M := utils.BuildKernelMatrix(cn.Conv1.Weight)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := utils.BuildBiasMatrix(cn.Conv1.Bias, inputLayerDim, batchSize)
	pool1M := utils.BuildKernelMatrix(cn.Pool1.Weight)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := utils.BuildBiasMatrix(cn.Pool1.Bias, inputLayerDim, batchSize)
	pool2M := utils.BuildKernelMatrix(cn.Pool2.Weight)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := utils.BuildBiasMatrix(cn.Pool2.Bias, inputLayerDim, batchSize)
	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	return weightMatrices, biasMatrices
}

//Rescale weights for approximation
func (cn *Cryptonet) RescaleForActivation(weights, biases []*mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	wRescaled := make([]*mat.Dense, len(weights))
	bRescaled := make([]*mat.Dense, len(biases))

	//skip last one as no activation
	for i := range weights[:len(weights)-1] {
		wRescaled[i], bRescaled[i] = cn.ReLUApprox.Rescale(weights[i], biases[i])
	}
	wRescaled[len(weights)-1], bRescaled[len(weights)-1] = weights[len(weights)-1], biases[len(weights)-1]
	return wRescaled, bRescaled
}

func (cn *Cryptonet) EncodeCryptonet(weights, biases []*mat.Dense, splits []cU.BlockSplits, Box cU.CkksBox, poolsize int) *CryptonetEcd {
	cne := new(CryptonetEcd)

	weightsR, biasesR := cn.RescaleForActivation(weights, biases)

	cne.Weights = make([]*cU.PlainWeightDiag, len(weightsR))
	cne.Bias = make([]*cU.PlainInput, len(biasesR))
	cne.Activators = make([]*cU.Activator, 2)

	level := Box.Params.MaxLevel()
	scale := Box.Params.DefaultScale()
	var err error

	leftInnerDim := splits[0].InnerRows
	inputRowP := splits[0].RowP

	iAct := 0

	for i, split := range splits[1:] {
		if i > 0 {
			//check masking for repacking
			if split.RowP != splits[i].ColP && splits[i].ColP%split.RowP != 0 {
				level--
			}
		}
		fmt.Println("Layer ", i+1, "level ", level)
		cne.Weights[i], err = cU.NewPlainWeightDiag(weightsR[i], split.RowP, split.ColP, leftInnerDim, level, Box)
		utils.ThrowErr(err)

		level-- //rescale

		cne.Bias[i], err = cU.NewPlainInput(biasesR[i], inputRowP, split.ColP, level, scale, Box)
		utils.ThrowErr(err)

		if iAct < 2 {
			cne.Activators[iAct], err = cU.NewActivator(cn.ReLUApprox, level, scale, leftInnerDim, split.InnerCols, Box, poolsize)
			utils.ThrowErr(err)
			iAct++
			level -= cn.ReLUApprox.LevelsOfAct()
		}
	}
	cne.Adder = cU.NewAdder(Box, poolsize)
	cne.Multiplier = cU.NewMultiplier(Box, poolsize)
	cne.Box = Box

	return cne
}

func (cn *Cryptonet) EncryptCryptonet(weights, biases []*mat.Dense, splits []cU.BlockSplits, Box cU.CkksBox, poolsize int) *CryptonetEnc {
	cne := new(CryptonetEnc)

	//compress layers 1 and 2 which are linear and back to back, then rescale for the activation
	weightsR, biasesR := cn.RescaleForActivation(weights, biases)

	cne.Weights = make([]*cU.EncWeightDiag, len(weights))
	cne.Bias = make([]*cU.EncInput, len(biases))
	cne.Activators = make([]*cU.Activator, 2)

	level := Box.Params.MaxLevel()
	scale := Box.Params.DefaultScale()
	var err error

	leftInnerDim := splits[0].InnerRows
	inputRowP := splits[0].RowP

	iAct := 0

	for i, split := range splits[1:] {
		if i > 0 {
			//check masking for repacking
			if split.RowP != splits[i].ColP && splits[i].ColP%split.RowP != 0 {
				level--
			}
		}
		fmt.Println("Layer ", i+1, "level ", level)
		cne.Weights[i], err = cU.NewEncWeightDiag(weightsR[i], split.RowP, split.ColP, leftInnerDim, level, Box)
		utils.ThrowErr(err)

		level-- //rescale

		cne.Bias[i], err = cU.NewEncInput(biasesR[i], inputRowP, split.ColP, level, scale, Box)
		utils.ThrowErr(err)

		if iAct < 2 {
			cne.Activators[iAct], err = cU.NewActivator(cn.ReLUApprox, level, scale, leftInnerDim, split.InnerCols, Box, poolsize)
			utils.ThrowErr(err)
			iAct++
			level -= cn.ReLUApprox.LevelsOfAct()
		}
	}
	cne.Adder = cU.NewAdder(Box, poolsize)
	cne.Multiplier = cU.NewMultiplier(Box, poolsize)
	cne.Box = Box

	return cne

}
