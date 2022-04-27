package utils

import (
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
	"math"
)

/*
	Define layer type for the various models
*/
type Bias struct {
	B   []float64 `json:"b"`
	Len int       `json:"len"`
}

type Kernel struct {
	/*
		Matrix M s.t X @ M = conv(X, layer).flatten() where X is a row-flattened data sample
		Clearly it can be generalized to a simple dense layer
	*/
	W    []float64 `json:"w"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

type Layer struct {
	Weight Kernel `json:"weight"`
	Bias   Bias   `json:"bias"`
}

type PolyApprox struct {
	Interval float64
	Degree   int
	Coeffs   []float64
}

func InitActivation(degree int, interval float64, coeffs []float64) PolyApprox {
	var activation PolyApprox
	activation.Degree = degree
	activation.Interval = interval
	activation.Coeffs = coeffs
	return activation
}

func InitReLU() PolyApprox {
	//use ReLU approximation: deg is 2 and inteval is [-10,10]
	var relu PolyApprox
	relu.Degree = 3
	relu.Interval = 10.0
	relu.Coeffs = make([]float64, relu.Degree)
	relu.Coeffs[0] = 1.1155
	relu.Coeffs[1] = 5
	relu.Coeffs[2] = 4.4003
	return relu
}

func BuildKernelMatrix(k Kernel) *mat.Dense {
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

func BuildBiasMatrix(b Bias, cols, batchSize int) *mat.Dense {
	// Compute a matrix containing the bias of the layer, to be added to the result of a Kernel multiplication
	res := mat.NewDense(batchSize, cols, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, plainUtils.Pad(b.B, cols-len(b.B)))
	}
	return res
}

func ActivatePlain(X *mat.Dense, activation PolyApprox) {
	/*
		Apply the activation function elementwise
	*/
	rows, cols := X.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := X.At(r, c) / float64(activation.Interval)
			res := 0.0
			for deg := 0; deg < activation.Degree; deg++ {
				res += (math.Pow(v, float64(deg)) * activation.Coeffs[deg])
			}
			X.Set(r, c, res)
		}
	}
}
