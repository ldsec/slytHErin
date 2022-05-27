package simpleNet

import "C"
import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"os"
	"time"
)

type SimpleNet struct {
	Conv1 utils.Layer `json:"conv1"`
	Pool1 utils.Layer `json:"pool1"`
	Pool2 utils.Layer `json:"pool2"`

	ReLUApprox utils.MinMaxPolyApprox //this will store the coefficients of the poly approximating ReLU
}
type SimpleNetPipeLine struct {
	//stores intermediate results of SimpleNetPipeline --> useful to be compared in encrypted pipeline
	OutConv1    *mat.Dense
	OutPool1    *mat.Dense
	OutPool2    *mat.Dense
	Predictions []int
	Corrects    int
	Time        time.Duration
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

func (sn *SimpleNet) Init() {
	deg := 3
	sn.ReLUApprox = utils.InitReLU(deg)
}

func (sn *SimpleNet) EvalBatchPlain(Xbatch [][]float64, Y []int, maxDim, labels int) *SimpleNetPipeLine {
	batchSize := len(Xbatch)
	Xflat := plainUtils.Vectorize(Xbatch, true) //tranpose true needed for now
	Xmat := mat.NewDense(len(Xbatch), len(Xbatch[0]), Xflat)

	var OutConv1 mat.Dense
	OutConv1.Mul(Xmat, utils.BuildKernelMatrix(sn.Conv1.Weight))
	_, cols := OutConv1.Dims()
	OutConv1.Add(&OutConv1, utils.BuildBiasMatrix(sn.Conv1.Bias, cols, batchSize))

	var OutPool1 mat.Dense
	OutPool1.Mul(&OutConv1, utils.BuildKernelMatrix(sn.Pool1.Weight))
	_, cols = OutPool1.Dims()
	OutPool1.Add(&OutPool1, utils.BuildBiasMatrix(sn.Pool1.Bias, cols, batchSize))
	utils.ActivatePlain(&OutPool1, sn.ReLUApprox)

	var OutPool2 mat.Dense
	OutPool2.Mul(&OutPool1, utils.BuildKernelMatrix(sn.Pool2.Weight))
	_, cols = OutPool2.Dims()
	OutPool2.Add(&OutPool2, utils.BuildBiasMatrix(sn.Pool2.Bias, cols, batchSize))
	utils.ActivatePlain(&OutPool2, sn.ReLUApprox)

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
	k1 := utils.BuildKernelMatrix(sn.Conv1.Weight)
	k1Blocks, err := plainUtils.PartitionMatrix(k1, 29, 65)
	if err != nil {
		panic(err)
	}
	C1, err := plainUtils.MultiPlyBlocks(XBlocks, k1Blocks)
	if err != nil {
		panic(err)
	}

	bias1 := utils.BuildBiasMatrix(sn.Conv1.Bias, C1.ColP*C1.InnerCols, C1.RowP*C1.InnerRows)
	bias1B, err := plainUtils.PartitionMatrix(bias1, C1.RowP, C1.ColP)
	if err != nil {
		panic(err)
	}
	C1b, err := plainUtils.AddBlocks(C1, bias1B)
	if err != nil {
		panic(err)
	}
	C1m := plainUtils.ExpandBlocks(C1b)
	//fmt.Println("____________Conv1__________________")
	//for i := 0; i < plainUtils.NumRows(C1m); i++ {
	//	fmt.Println("Test:", C1m.RawRowView(i))
	//	fmt.Println("Expected:", normalRes.OutConv1.RawRowView(i))
	//}
	//fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(C1m), plainUtils.RowFlatten(normalRes.OutConv1)))

	pool1Blocks, err := plainUtils.PartitionMatrix(utils.BuildKernelMatrix(sn.Pool1.Weight), C1.ColP, 10)
	C2, err := plainUtils.MultiPlyBlocks(C1b, pool1Blocks)
	if err != nil {
		panic(err)
	}
	bias2B, err := plainUtils.PartitionMatrix(utils.BuildBiasMatrix(sn.Pool1.Bias, C2.ColP*C2.InnerCols, C2.RowP*C2.InnerRows), C2.RowP, C2.ColP)
	C2, err = plainUtils.AddBlocks(C2, bias2B)
	for i := range C2.Blocks {
		for j := range C2.Blocks[i] {
			utils.ActivatePlain(C2.Blocks[i][j], sn.ReLUApprox)
		}
	}

	C2m := plainUtils.ExpandBlocks(C2)
	//fmt.Println("____________Pool1__________________")
	//for i := 0; i < plainUtils.NumRows(C2m); i++ {
	//	fmt.Println("Test:", C2m.RawRowView(i))
	//	fmt.Println("Expected:", normalRes.OutPool1.RawRowView(i))
	//}
	//fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(C2m), plainUtils.RowFlatten(normalRes.OutPool1)))

	pool2Blocks, err := plainUtils.PartitionMatrix(utils.BuildKernelMatrix(sn.Pool2.Weight), C2.ColP, 1)
	C3, err := plainUtils.MultiPlyBlocks(C2, pool2Blocks)
	if err != nil {
		panic(err)
	}
	bias3B, err := plainUtils.PartitionMatrix(utils.BuildBiasMatrix(sn.Pool2.Bias, C3.ColP*C3.InnerCols, C3.RowP*C3.InnerRows), C3.RowP, C3.ColP)
	C3b, err := plainUtils.AddBlocks(C3, bias3B)
	for i := range C3b.Blocks {
		for j := range C3b.Blocks[i] {
			utils.ActivatePlain(C3b.Blocks[i][j], sn.ReLUApprox)
		}
	}
	//fmt.Println("Rows:", C3.RowP)
	//fmt.Println("Cols:", C3.ColP)
	//fmt.Println("Sub-Rows:", C3.InnerRows)
	//fmt.Println("Sub-Cols:", C3.InnerCols)
	res := plainUtils.ExpandBlocks(C3b)
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

func (sn *SimpleNet) EvalBatchEncrypted(XBatchClear [][]float64, Y []int, XbatchEnc *cipherUtils.EncInput, weightsBlock []*cipherUtils.PlainWeightDiag, biasBlock []*cipherUtils.PlainInput, Box cipherUtils.CkksBox, labels int) *SimpleNetPipeLine {

	//code for debug...

	plainResults := sn.EvalBatchPlainBlocks(XBatchClear, Y, 10)
	fmt.Println("Loaded plaintext results")

	//pipeline
	now := time.Now()

	fmt.Println("Conv1", weightsBlock[0].Blocks[0][0].Diags[0].Level())
	A, err := cipherUtils.BlocksC2PMul(XbatchEnc, weightsBlock[0], Box)
	utils.ThrowErr(err)
	fmt.Println("Adding bias")
	Ab, err := cipherUtils.AddBlocksC2P(A, biasBlock[0], Box)
	utils.ThrowErr(err)
	exp, err := plainUtils.PartitionMatrix(plainResults.OutConv1, Ab.RowP, Ab.ColP)
	utils.ThrowErr(err)
	cipherUtils.CompareBlocks(Ab, exp, Box)
	cipherUtils.PrintDebugBlocks(Ab, exp, Box)

	fmt.Println("Pool1", weightsBlock[1].Blocks[0][0].Diags[0].Level())
	B, err := cipherUtils.BlocksC2PMul(Ab, weightsBlock[1], Box)
	fmt.Println("Adding bias")
	utils.ThrowErr(err)
	Bb, err := cipherUtils.AddBlocksC2P(B, biasBlock[1], Box)
	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Bb, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)
	exp, err = plainUtils.PartitionMatrix(plainResults.OutPool1, Bb.RowP, Bb.ColP)
	utils.ThrowErr(err)
	cipherUtils.CompareBlocks(Bb, exp, Box)
	cipherUtils.PrintDebugBlocks(Bb, exp, Box)

	fmt.Println("Pool2")
	BbF, err := cipherUtils.NewEncInput(cipherUtils.DecInput(Bb, Box), Bb.RowP, Bb.ColP, 7, Box)
	CC, err := cipherUtils.BlocksC2PMul(Bb, weightsBlock[2], Box)
	CF, err := cipherUtils.BlocksC2PMul(BbF, weightsBlock[2], Box)
	pool2Blocks, err := plainUtils.PartitionMatrix(utils.BuildKernelMatrix(sn.Pool2.Weight), weightsBlock[2].RowP, weightsBlock[2].ColP)
	C2, _ := plainUtils.PartitionMatrix(plainResults.OutPool1, Bb.RowP, Bb.ColP)
	C3, err := plainUtils.MultiPlyBlocks(C2, plainUtils.MultiplyBlocksByConst(pool2Blocks, 0.1))
	fmt.Println("Enc vs plain")

	cipherUtils.CompareBlocks(CC, C3, Box)
	cipherUtils.PrintDebugBlocks(CC, C3, Box)
	fmt.Println("Enc fresh vs plain")
	cipherUtils.CompareBlocks(CF, C3, Box)
	cipherUtils.PrintDebugBlocks(CF, C3, Box)
	if err != nil {
		panic(err)
	}

	fmt.Println("Adding bias")
	utils.ThrowErr(err)
	Cb, err := cipherUtils.AddBlocksC2P(CC, biasBlock[2], Box)
	CbF, err := cipherUtils.AddBlocksC2P(CF, biasBlock[2], Box)
	utils.ThrowErr(err)
	bias3B, err := plainUtils.PartitionMatrix(utils.BuildBiasMatrix(sn.Pool2.Bias, C3.ColP*C3.InnerCols, C3.RowP*C3.InnerRows), C3.RowP, C3.ColP)
	C3b, err := plainUtils.AddBlocks(C3, plainUtils.MultiplyBlocksByConst(bias3B, 0.1))
	fmt.Println("Enc vs plain")
	cipherUtils.CompareBlocks(Cb, C3b, Box)
	fmt.Println("Enc fresh vs plain")
	cipherUtils.CompareBlocks(CbF, C3b, Box)

	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Cb, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)
	cipherUtils.EvalPolyBlocks(CbF, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)
	plainUtils.MultiplyBlocksByConst(C3b, 10.0) //it gets divided in activation
	for i := range C3b.Blocks {
		for j := range C3b.Blocks[i] {
			utils.ActivatePlain(C3b.Blocks[i][j], sn.ReLUApprox)
		}
	}
	fmt.Println("Enc vs plain")
	cipherUtils.CompareBlocks(Cb, C3b, Box)
	fmt.Println("Enc fresh vs plain")
	cipherUtils.CompareBlocks(CbF, C3b, Box)
	fmt.Println("______________________")
	fmt.Println("Done", time.Since(now))

	exp, _ = plainUtils.PartitionMatrix(plainResults.OutPool2, Cb.RowP, Cb.ColP)
	cipherUtils.CompareBlocks(Cb, exp, Box)
	cipherUtils.PrintDebugBlocks(Cb, exp, Box)
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
	fmt.Println("Corrects enc:", corrects)
	fmt.Println("Corrects clear:", plainResults.Corrects)

	outConv1 := cipherUtils.DecInput(Ab, Box)
	outPool1 := cipherUtils.DecInput(Bb, Box)
	outPool2 := cipherUtils.DecInput(Cb, Box)

	return &SimpleNetPipeLine{
		OutConv1:    plainUtils.NewDense(outConv1),
		OutPool1:    plainUtils.NewDense(outPool1),
		OutPool2:    plainUtils.NewDense(outPool2),
		Predictions: predictions,
		Corrects:    corrects,
	}
}

func (sn *SimpleNet) EvalBatchEncryptedCompressed(XBatchClear [][]float64, Y []int, XbatchEnc *cipherUtils.EncInput, weightsBlock []*cipherUtils.PlainWeightDiag, biasBlock []*cipherUtils.PlainInput, Box cipherUtils.CkksBox, labels int, debug bool) *SimpleNetPipeLine {
	plainResults := sn.EvalBatchPlainBlocks(XBatchClear, Y, 10)
	fmt.Println("Loaded plaintext results")
	//pipeline
	now := time.Now()
	//fmt.Println("Compressed conv + pool", weightsBlock[0].Blocks[0][0].Diags[0].Level())
	A, err := cipherUtils.BlocksC2PMul(XbatchEnc, weightsBlock[0], Box)
	utils.ThrowErr(err)

	fmt.Println("Adding bias")
	Ab, err := cipherUtils.AddBlocksC2P(A, biasBlock[0], Box)

	utils.ThrowErr(err)
	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Ab, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)
	if debug {
		exp, err := plainUtils.PartitionMatrix(plainResults.OutPool1, Ab.RowP, Ab.ColP)
		utils.ThrowErr(err)
		cipherUtils.PrintDebugBlocks(Ab, exp, Box)
		cipherUtils.CompareBlocks(Ab, exp, Box)
	}
	fmt.Println("Pool2")
	CC, err := cipherUtils.BlocksC2PMul(Ab, weightsBlock[1], Box)
	utils.ThrowErr(err)
	if err != nil {
		panic(err)
	}
	fmt.Println("Adding bias")
	utils.ThrowErr(err)
	Cb, err := cipherUtils.AddBlocksC2P(CC, biasBlock[1], Box)
	utils.ThrowErr(err)
	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Cb, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)
	fmt.Println("______________________")
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)
	if debug {
		exp, _ := plainUtils.PartitionMatrix(plainResults.OutPool2, Cb.RowP, Cb.ColP)
		cipherUtils.CompareBlocks(Cb, exp, Box)
		cipherUtils.PrintDebugBlocks(Cb, exp, Box)
	}
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
	fmt.Println("Corrects enc:", corrects)
	fmt.Println("Corrects clear:", plainResults.Corrects)

	outConv1 := cipherUtils.DecInput(Ab, Box)
	outPool2 := cipherUtils.DecInput(Cb, Box)

	return &SimpleNetPipeLine{
		OutConv1:    plainUtils.NewDense(outConv1),
		OutPool1:    nil,
		OutPool2:    plainUtils.NewDense(outPool2),
		Predictions: predictions,
		Corrects:    corrects,
		Time:        elapsed,
	}
}

func (sn *SimpleNet) EvalBatchEncryptedCompressed_Light(Y []int, XbatchEnc *cipherUtils.EncInput, weightsBlock []*cipherUtils.PlainWeightDiag, biasBlock []*cipherUtils.PlainInput, Box cipherUtils.CkksBox, labels int) *SimpleNetPipeLine {
	fmt.Println("Starting...")
	//pipeline
	now := time.Now()
	//fmt.Println("Compressed conv + pool", weightsBlock[0].Blocks[0][0].Diags[0].Level())
	A, err := cipherUtils.BlocksC2PMul(XbatchEnc, weightsBlock[0], Box)
	utils.ThrowErr(err)

	fmt.Println("Adding bias")
	Ab, err := cipherUtils.AddBlocksC2P(A, biasBlock[0], Box)

	utils.ThrowErr(err)
	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Ab, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)

	fmt.Println("Pool2")
	CC, err := cipherUtils.BlocksC2PMul(Ab, weightsBlock[1], Box)
	utils.ThrowErr(err)
	if err != nil {
		panic(err)
	}
	fmt.Println("Adding bias")
	utils.ThrowErr(err)
	Cb, err := cipherUtils.AddBlocksC2P(CC, biasBlock[1], Box)
	utils.ThrowErr(err)
	fmt.Println("Activation")
	cipherUtils.EvalPolyBlocks(Cb, ckks.NewPoly(plainUtils.RealToComplex(sn.ReLUApprox.Coeffs)), Box)
	fmt.Println("______________________")
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	batchSize := len(Y)
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
	fmt.Println("Corrects :", corrects)

	outConv1 := cipherUtils.DecInput(Ab, Box)
	outPool2 := cipherUtils.DecInput(Cb, Box)

	return &SimpleNetPipeLine{
		OutConv1:    plainUtils.NewDense(outConv1),
		OutPool1:    nil,
		OutPool2:    plainUtils.NewDense(outPool2),
		Predictions: predictions,
		Corrects:    corrects,
		Time:        elapsed,
	}
}
