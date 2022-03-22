package modelsPlain

import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math"
	"os"
)

type Bias struct {
	B   []float64 `json:"b"`
	Len int       `json:"len"`
}
type Channel struct {
	W    []float64 `json:"w"` //this should be the row-flattening of the tranpsosed kernel matrix
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}
type Kernel struct {
	Channels []Channel `json:"channels"`
}

type ConvLayer struct {
	Weight                                               []Kernel `json:"weight"`
	Bias                                                 Bias     `json:"bias"`
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

func convolutionPlain(layer ConvLayer, Xmat *mat.Dense) []*mat.Dense {
	OutConv := make([]*mat.Dense, layer.outChans)
	for i := 0; i < layer.outChans; i++ {
		//for every kernel
		var kernelRes *mat.Dense
		for j := 0; j < layer.inChans; j++ {
			channel := layer.Weight[i].Channels[j]
			channelMat := mat.NewDense(channel.Rows, channel.Cols, channel.W)
			var res *mat.Dense
			res.Mul(Xmat, channelMat)
			kernelRes.Add(kernelRes, res)
		}
		rows, cols := kernelRes.Dims()
		biasFlat := make([]float64, rows*cols)
		for k := 0; k < rows*cols; k++ {
			biasFlat[k] = layer.Bias.B[i]
		}
		biasMat := mat.NewDense(rows, cols, biasFlat)
		kernelRes.Add(kernelRes, biasMat)
		OutConv[i] = kernelRes
	}
	return OutConv
}

func poolElemWisePlain(pool ConvLayer, X []*mat.Dense) []*mat.Dense {
	OutPool := make([]*mat.Dense, pool.outChans)
	for i := 0; i < pool.outChans; i++ {
		var kernelRes float64
		for j := 0; j < pool.inChans; j++ {
			kernelChannel := mat.NewDense(pool.Weight[i].Channels[j].Rows,
				pool.Weight[i].Channels[j].Cols,
				pool.Weight[i].Channels[j].W)
			inputChannel := OutPool[i]
			var res *mat.Dense
			res.MulElem(inputChannel, kernelChannel)
			kernelRes += mat.Sum(res)
		}
		kernelRes += pool.Bias.B[i]
		OutPool[i] = mat.NewDense(1, 1, []float64{kernelRes})
	}
	return OutPool
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

	sn.Pool2.kernelSize = 10
	sn.Pool2.inDim = 13
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
	sn.ReLUApprox.Coeffs[2] = 4.003
}

func (sn *SimpleNet) ActivatePlain(X []*mat.Dense) {
	/*
		Apply the activation function elementwise
	*/
	for i := range X {
		x := X[i]
		rows, cols := x.Dims()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				v := x.At(r, c) / float64(sn.ReLUApprox.Interval)
				res := 0.0
				for deg := 0; deg < sn.ReLUApprox.Degree; deg++ {
					res += (math.Pow(v, float64(deg)) * sn.ReLUApprox.Coeffs[deg])
				}
				x.Set(r, c, res)
			}
		}
	}
}

func (sn *SimpleNet) EvalBatchPlain(Xbatch [][]float64, Y [][]float64) {
	Xflat := plainUtils.Vectorize(Xbatch, false)
	Xmat := mat.NewDense(len(Xbatch), len(Xbatch[0]), Xflat)

	//Conv1
	OutConv1 := convolutionPlain(sn.Conv1, Xmat)
	sn.ActivatePlain(OutConv1) //check if this is by reference

	//Pool1 as elementwise
	OutPool1 := poolElemWisePlain(sn.Pool1, OutConv1)
	sn.ActivatePlain(OutPool1)

	OutPool2 := poolElemWisePlain(sn.Pool2, OutPool1)
	r, c := OutPool2[0].Dims()
	fmt.Println(len(OutPool2), r, c)
}
