package modelsPlain

import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"io/ioutil"
	"math"
	"os"
	"reflect"
	"sync"

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
type SimpleNetPipeLine struct {
	//stores intermediate results of SimpleNetPipeline --> useful to be compared in encrypted pipeline
	OutConv1    *mat.Dense
	OutPool1    *mat.Dense
	OutPool2    *mat.Dense
	Predictions []int
	Corrects    int
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

func buildKernelMatrix(k Kernel, dimension int) *mat.Dense {
	/*
		Returns a matrix M s.t X.M = conv(x,layer)
		kernel matrix is in a square form with even dimensions to apply the complex trick
		Reference: pg.3 of https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
	*/
	if dimension == -1 {
		//means this if the layer we should extrapolate the max dimension of the network
		if k.Rows > k.Cols {
			dimension = k.Rows
		} else {
			dimension = k.Cols
		}
		for (dimension % 2) != 0 {
			dimension++
		}
	}
	res := mat.NewDense(dimension, dimension, nil)
	for i := 0; i < k.Rows; i++ {
		for j := 0; j < k.Cols; j++ {
			res.Set(i, j, k.W[i*k.Cols+j])
		}
	}
	return res
}

func buildBiasMatrix(b Bias, cols, batchSize int) *mat.Dense {
	// Compute a matrix containing the bias of the layer, to be added to the result
	res := mat.NewDense(batchSize, cols, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, plainUtils.Pad(b.B, cols-len(b.B)))
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

func (sn *SimpleNet) EvalBatchPlain(Xbatch [][]float64, Y []int, maxDim, labels int) *SimpleNetPipeLine {
	batchSize := len(Xbatch)
	Xflat := plainUtils.Vectorize(Xbatch, true) //tranpose true needed for now
	Xmat := mat.NewDense(len(Xbatch), len(Xbatch[0]), Xflat)

	var OutConv1 mat.Dense
	OutConv1.Mul(Xmat, buildKernelMatrix(sn.Conv1.Weight, -1))
	_, cols := OutConv1.Dims()
	OutConv1.Add(&OutConv1, buildBiasMatrix(sn.Conv1.Bias, cols, batchSize))

	var OutPool1 mat.Dense
	OutPool1.Mul(&OutConv1, buildKernelMatrix(sn.Pool1.Weight, maxDim))
	_, cols = OutPool1.Dims()
	OutPool1.Add(&OutPool1, buildBiasMatrix(sn.Pool1.Bias, cols, batchSize))
	sn.ActivatePlain(&OutPool1)

	var OutPool2 mat.Dense
	OutPool2.Mul(&OutPool1, buildKernelMatrix(sn.Pool2.Weight, maxDim))
	cols, cols = OutPool2.Dims()
	OutPool2.Add(&OutPool2, buildBiasMatrix(sn.Pool2.Bias, cols, batchSize))
	sn.ActivatePlain(&OutPool2)

	predictions := make([]int, batchSize)
	corrects := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
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
	return &SimpleNetPipeLine{
		OutConv1:    &OutConv1,
		OutPool1:    &OutPool1,
		OutPool2:    &OutPool2,
		Predictions: predictions,
		Corrects:    corrects,
	}
}

func (sn *SimpleNet) EvalBatchEncrypted(XBatchClear [][]float64, XbatchEnc *ckks.Ciphertext, weightMatrices, biasMatrices []*mat.Dense, Box cipherUtils.CkksBox, maxDim int, glassDoor bool) *ckks.Ciphertext {
	var plainResults *SimpleNetPipeLine
	if glassDoor {
		plainResults = sn.EvalBatchPlain(XBatchClear, make([]int, len(XBatchClear)), maxDim, 10)
	}

	var wg sync.WaitGroup
	weights := make([][][]complex128, len(weightMatrices))
	for i := 0; i < len(weightMatrices); i++ {
		wg.Add(1)
		go func(weights [][][]complex128, i int) {
			defer wg.Done()
			weights[i] = cipherUtils.FormatWeights(plainUtils.MatToArray(weightMatrices[i]), len(XBatchClear))
		}(weights, i)
		//weights[i] = cipherUtils.FormatWeights(plainUtils.MatToArray(weightMatrices[i]), maxDim)
	}
	wg.Wait()
	//encode weights and bias as Plaintext:
	//each plainW contains an array of Plaintext, each containing one diagonal of the kernel matrix
	//plainB contains the flattened bias matrices
	plainW := make([][]*ckks.Plaintext, len(weights))
	plainB := make([]*ckks.Plaintext, len(biasMatrices))
	for i := range weights {
		wg.Add(1)
		go func(plainW [][]*ckks.Plaintext, plainB []*ckks.Plaintext, i int) {
			defer wg.Done()
			pt := ckks.NewPlaintext(Box.Params, Box.Params.MaxLevel(), Box.Params.QiFloat64(Box.Params.MaxLevel()))
			for j := range weights[0] {
				//EncodeSlots puts the values in the plaintext in a way
				//such that homomorphic elem-wise mult is preserved
				Box.Encoder.EncodeSlots(weights[i][j], pt, Box.Params.LogSlots())
				plainW[i][j] = pt
			}
			Box.Encoder.EncodeSlots(plainUtils.Vectorize(plainUtils.MatToArray(biasMatrices[i]), true), pt, Box.Params.LogSlots())
			plainB[i] = pt
		}(plainW, plainB, i)
	}
	wg.Wait()
	for i := range plainW {
		resCt := cipherUtils.Cipher2PMul(XbatchEnc, len(XBatchClear), maxDim, plainW[i], true, true, Box)
		resCt = Box.Evaluator.AddNew(resCt, plainB[i])
		if glassDoor {
			//trick to loop
			v := reflect.ValueOf(plainResults)
			resPlain := v.Field(i).Interface().(*mat.Dense)
			resPlainRf := plainUtils.RowFlatten(resPlain)
			resPlainRfC := plainUtils.RealToComplex(resPlainRf)
			cipherUtils.PrintDebug(resCt, resPlainRfC, Box)
		}
	}
	return nil
}
