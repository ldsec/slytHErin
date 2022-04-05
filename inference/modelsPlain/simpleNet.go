package modelsPlain

import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math"
	"os"
	"time"
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

func buildKernelMatrix(k Kernel) *mat.Dense {
	/*
		Returns a matrix M s.t X.M = conv(x,layer)
	*/

	res := mat.NewDense(k.Rows, k.Cols, nil)
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
	OutConv1.Mul(Xmat, buildKernelMatrix(sn.Conv1.Weight))
	_, cols := OutConv1.Dims()
	OutConv1.Add(&OutConv1, buildBiasMatrix(sn.Conv1.Bias, cols, batchSize))

	var OutPool1 mat.Dense
	OutPool1.Mul(&OutConv1, buildKernelMatrix(sn.Pool1.Weight))
	_, cols = OutPool1.Dims()
	OutPool1.Add(&OutPool1, buildBiasMatrix(sn.Pool1.Bias, cols, batchSize))
	sn.ActivatePlain(&OutPool1)

	var OutPool2 mat.Dense
	OutPool2.Mul(&OutPool1, buildKernelMatrix(sn.Pool2.Weight))
	_, cols = OutPool2.Dims()
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

func (sn *SimpleNet) EvalBatchPlainBlocks(Xbatch [][]float64, Y []int, labels int) *SimpleNetPipeLine {
	//evaluate batch using Block Matrix arithmetics for efficient computations with small ciphers
	batchSize := len(Xbatch)
	Xflat := plainUtils.Vectorize(Xbatch, true) //tranpose true needed for now
	Xmat := mat.NewDense(len(Xbatch), len(Xbatch[0]), Xflat)
	//normalRes := sn.EvalBatchPlain(Xbatch, Y, 0, labels)
	XBlocks, err := plainUtils.PartitionMatrix(Xmat, batchSize, 29)
	if err != nil {
		panic(err)
	}
	k1 := buildKernelMatrix(sn.Conv1.Weight)
	k1Blocks, err := plainUtils.PartitionMatrix(k1, 29, 5)
	if err != nil {
		panic(err)
	}
	C1, err := plainUtils.MultiPlyBlocks(XBlocks, k1Blocks)
	if err != nil {
		panic(err)
	}

	bias1 := buildBiasMatrix(sn.Conv1.Bias, C1.ColP*C1.InnerCols, C1.RowP*C1.InnerRows)
	bias1B, err := plainUtils.PartitionMatrix(bias1, C1.RowP, C1.ColP)
	if err != nil {
		panic(err)
	}
	C1, err = plainUtils.AddBlocks(C1, bias1B)
	if err != nil {
		panic(err)
	}
	C1m := plainUtils.ExpandBlocks(C1)
	//fmt.Println("____________Conv1__________________")
	//for i := 0; i < plainUtils.NumRows(C1m); i++ {
	//	fmt.Println("Test:", C1m.RawRowView(i))
	//	fmt.Println("Expected:", normalRes.OutConv1.RawRowView(i))
	//}
	//fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(C1m), plainUtils.RowFlatten(normalRes.OutConv1)))

	pool1Blocks, err := plainUtils.PartitionMatrix(buildKernelMatrix(sn.Pool1.Weight), C1.ColP, 10)
	C2, err := plainUtils.MultiPlyBlocks(C1, pool1Blocks)
	if err != nil {
		panic(err)
	}
	bias2B, err := plainUtils.PartitionMatrix(buildBiasMatrix(sn.Pool1.Bias, C2.ColP*C2.InnerCols, C2.RowP*C2.InnerRows), C2.RowP, C2.ColP)
	C2, err = plainUtils.AddBlocks(C2, bias2B)
	for i := range C2.Blocks {
		for j := range C2.Blocks[i] {
			sn.ActivatePlain(C2.Blocks[i][j])
		}
	}

	C2m := plainUtils.ExpandBlocks(C2)
	//fmt.Println("____________Pool1__________________")
	//for i := 0; i < plainUtils.NumRows(C2m); i++ {
	//	fmt.Println("Test:", C2m.RawRowView(i))
	//	fmt.Println("Expected:", normalRes.OutPool1.RawRowView(i))
	//}
	//fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(C2m), plainUtils.RowFlatten(normalRes.OutPool1)))

	pool2Blocks, err := plainUtils.PartitionMatrix(buildKernelMatrix(sn.Pool2.Weight), C2.ColP, 1)
	C3, err := plainUtils.MultiPlyBlocks(C2, pool2Blocks)
	if err != nil {
		panic(err)
	}
	bias3B, err := plainUtils.PartitionMatrix(buildBiasMatrix(sn.Pool2.Bias, C3.ColP*C3.InnerCols, C3.RowP*C3.InnerRows), C3.RowP, C3.ColP)
	C3, err = plainUtils.AddBlocks(C3, bias3B)
	for i := range C3.Blocks {
		for j := range C3.Blocks[i] {
			sn.ActivatePlain(C3.Blocks[i][j])
		}
	}
	//fmt.Println("Rows:", C3.RowP)
	//fmt.Println("Cols:", C3.ColP)
	//fmt.Println("Sub-Rows:", C3.InnerRows)
	//fmt.Println("Sub-Cols:", C3.InnerCols)
	res := plainUtils.ExpandBlocks(C3)
	//fmt.Println("____________Conv2__________________")
	//for i := 0; i < plainUtils.NumRows(res); i++ {
	//	fmt.Println("Test:", res.RawRowView(i))
	//	fmt.Println("Expected:", normalRes.OutPool1.RawRowView(i))
	//}
	//fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(res), plainUtils.RowFlatten(normalRes.OutPool2)))

	predictions := make([]int, batchSize)
	corrects := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := res.At(i, j)
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
		OutConv1:    C1m,
		OutPool1:    C2m,
		OutPool2:    res,
		Predictions: predictions,
		Corrects:    corrects,
	}
}

//TO DO: implement this with encrypted block matrix logic --> encBlocks.go and opsBlocks.go of package cipherUtils
func (sn *SimpleNet) EvalBatchEncrypted(XBatchClear [][]float64, Y []int, XbatchEnc *cipherUtils.EncInput, weightsBlock []*cipherUtils.PlainWeightDiag, biasBlock []*cipherUtils.PlainInput, Box cipherUtils.CkksBox, labels int) *SimpleNetPipeLine {

	plainResults := sn.EvalBatchPlainBlocks(XBatchClear, Y, 10)
	fmt.Println("Loaded plaintext results")
	//build weights and bias in block forms. Also multiply by interval to spare a level during activation

	//pipeline
	now := time.Now()
	fmt.Println("Conv1")
	A, err := cipherUtils.BlocksC2PMul(XbatchEnc, weightsBlock[0], Box)
	utils.ThrowErr(err)
	fmt.Println("Done:", time.Since(now))
	now = time.Now()
	fmt.Println("Adding bias")
	Ab, err := cipherUtils.AddBlocksC2P(A, biasBlock[0], Box)
	fmt.Println("Done:", time.Since(now))
	exp, _ := plainUtils.PartitionMatrix(plainResults.OutConv1, Ab.RowP, Ab.ColP)
	cipherUtils.CompareBlocks(Ab, exp, Box)

	now = time.Now()
	fmt.Println("Pool1")
	B, err := cipherUtils.BlocksC2PMul(Ab, weightsBlock[1], Box)
	fmt.Println("Done:", time.Since(now))
	now = time.Now()
	fmt.Println("Adding bias")
	utils.ThrowErr(err)
	Bb, err := cipherUtils.AddBlocksC2P(B, biasBlock[1], Box)
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Bb, sn.ReLUApprox.Coeffs, Box)
	fmt.Println("Done:", time.Since(now))
	now = time.Now()
	fmt.Println("Bootstrapping")
	cipherUtils.BootStrapBlocks(Bb, Box)
	fmt.Println("Done:", time.Since(now))
	exp, _ = plainUtils.PartitionMatrix(plainResults.OutPool1, Bb.RowP, Bb.ColP)
	cipherUtils.CompareBlocks(Bb, exp, Box)

	now = time.Now()
	fmt.Println("Pool2")
	C, err := cipherUtils.BlocksC2PMul(Bb, weightsBlock[1], Box)
	fmt.Println("Done:", time.Since(now))
	now = time.Now()
	fmt.Println("Adding bias")
	utils.ThrowErr(err)
	Cb, err := cipherUtils.AddBlocksC2P(C, biasBlock[1], Box)
	fmt.Println("Done:", time.Since(now))

	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Cb, sn.ReLUApprox.Coeffs, Box)
	fmt.Println("Done:", time.Since(now))
	exp, _ = plainUtils.PartitionMatrix(plainResults.OutPool2, Cb.RowP, Cb.ColP)
	cipherUtils.CompareBlocks(Cb, exp, Box)
	//cipherUtils.BootStrapBlocks(Cb, Box)
	batchSize := len(XBatchClear)
	res := cipherUtils.DecInput(Cb, Box)
	predictions := make([]int, batchSize)
	corrects := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := res[i][j]
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
		OutConv1:    nil,
		OutPool1:    nil,
		OutPool2:    nil,
		Predictions: predictions,
		Corrects:    corrects,
	}
}
