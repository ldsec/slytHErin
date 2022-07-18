package utils

import (
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
	"time"
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

type Stats struct {
	Predictions []int
	Corrects    int
	Accuracy    float64
	Time        time.Duration
}

// Returns a matrix M s.t X x M = conv(x,layer), or a dense layer
func BuildKernelMatrix(k Kernel) *mat.Dense {

	res := mat.NewDense(k.Rows, k.Cols, nil)
	for i := 0; i < k.Rows; i++ {
		for j := 0; j < k.Cols; j++ {
			res.Set(i, j, k.W[i*k.Cols+j])
		}
	}
	return res
}

// Compute a matrix containing the bias of the layer, to be added to the result of a Kernel multiplication
func BuildBiasMatrix(b Bias, cols, batchSize int) *mat.Dense {
	res := mat.NewDense(batchSize, cols, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, plainUtils.Pad(b.B, cols-len(b.B)))
	}
	return res
}

func Predict(Y []int, labels int, result [][]float64) (int, float64, []int) {
	batchSize := len(Y)
	predictions := make([]int, batchSize)
	corrects := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := result[i][j]
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
	accuracy := float64(corrects) / float64(batchSize)
	return corrects, accuracy, predictions
}
