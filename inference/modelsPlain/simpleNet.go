package modelsPlain

import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
)

type Bias struct {
	B   []float64 `json:"b"`
	Len int       `json:"len"`
}

type Kernel struct {
	W    []float64 `json:"w"` //Matrix M s.t X @ M = conv(X, layer).flatten() where X is a row-flattened data sample
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

type ConvLayer struct {
	Weight                                               Kernel `json:"weight"`
	Bias                                                 Bias   `json:"bias"`
	kernelSize, inChans, outChans, stride, inDim, outDim int
}
type PolyApprox struct {
	Interval, Degree int
	Coeffs           []float64
}
type SimpleNet struct {
	Conv1 ConvLayer `json:"conv1"`
	Pool1 ConvLayer `json:"pool1"`
	Pool2 ConvLayer `json:"pool2"`

	ReLUApprox PolyApprox //this will store the coefficients of the poly approximating ReLU
}

/***************************
HELPERS
 ***************************/
func LoadSimpleNet(path string) *SimpleNet {
	// loads json file with weights
	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res SimpleNet
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}

/****************
SIMPLENET METHODS
 ***************/

func (sn *SimpleNet) InitDim() {
	sn.Conv1.kernelSize = 5
	sn.Conv1.inDim = 29
	sn.Conv1.outDim = 13
	sn.Conv1.inChans = 1
	sn.Conv1.outChans = 5

	sn.Pool1.kernelSize = 13
	sn.Pool1.inDim = 13
	sn.Pool1.outDim = 1
	sn.Pool1.inChans = 5    //#filters per kernel
	sn.Pool1.outChans = 100 //#kernels

	sn.Pool2.kernelSize = 1
	sn.Pool2.inDim = 1
	sn.Pool2.outDim = 1
	sn.Pool2.inChans = 100 //#filters per kernel
	sn.Pool2.outChans = 10 //#kernels
}

func (sn *SimpleNet) InitActivation() {
	sn.ReLUApprox.Degree = 3
	sn.ReLUApprox.Interval = 10
	sn.ReLUApprox.Coeffs = make([]float64, sn.ReLUApprox.Degree)
	sn.ReLUApprox.Coeffs[0] = 1.1155
	sn.ReLUApprox.Coeffs[1] = 5
	sn.ReLUApprox.Coeffs[2] = 4.4003
}

func buildKernelMatrix(k Kernel) *mat.Dense {
	return mat.NewDense(k.Rows, k.Cols, k.W)
}

func buildBiasMatrix(b Bias, batchSize int) *mat.Dense {
	res := mat.NewDense(batchSize, b.Len, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, b.B)
	}
	return res
}

func (sn *SimpleNet) ActivatePlain(X *mat.Dense) {
	/*
		Apply the activation function elementwise
	*/
	rows, cols := X.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := X.At(r, c) / float64(sn.ReLUApprox.Interval)
			res := 0.0
			for deg := 0; deg < sn.ReLUApprox.Degree; deg++ {
				res += (math.Pow(v, float64(deg)) * sn.ReLUApprox.Coeffs[deg])
			}
			X.Set(r, c, res)
		}
	}
}

func (sn *SimpleNet) EvalBatchPlain(Xbatch [][]float64, Y []int) int {
	batchSize := len(Xbatch)
	Xflat := plainUtils.Vectorize(Xbatch, true) //tranpose true needed for now
	Xmat := mat.NewDense(len(Xbatch), len(Xbatch[0]), Xflat)

	var OutConv1 mat.Dense
	OutConv1.Mul(Xmat, buildKernelMatrix(sn.Conv1.Weight))
	OutConv1.Add(&OutConv1, buildBiasMatrix(sn.Conv1.Bias, batchSize))

	var OutPool1 mat.Dense
	OutPool1.Mul(&OutConv1, buildKernelMatrix(sn.Pool1.Weight))
	OutPool1.Add(&OutPool1, buildBiasMatrix(sn.Pool1.Bias, batchSize))
	sn.ActivatePlain(&OutPool1)

	var OutPool2 mat.Dense
	OutPool2.Mul(&OutPool1, buildKernelMatrix(sn.Pool2.Weight))
	OutPool2.Add(&OutPool2, buildBiasMatrix(sn.Pool2.Bias, batchSize))
	sn.ActivatePlain(&OutPool2)
	predictions := make([]int, batchSize)
	corrects := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < 10; j++ {
			confidence := OutPool2.At(i, j)
			if confidence > maxConfidence {
				maxConfidence = confidence
				maxIdx = j
			}
		}
		predictions[i] = maxIdx
		if predictions[i] == Y[i] {
			corrects += 1
		}
	}
	return corrects
}
