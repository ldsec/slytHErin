package utils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
)

/*
	Define layer type for the various models
*/
type Bias struct {
	B   []float64 `json:"b"`
	Len int       `json:"len"`
}

/*
	Matrix M s.t X @ M = conv(X, layer).flatten() where X is a row-flattened data sample
	Clearly it can be generalized to a simple dense layer
*/
type Kernel struct {
	W    []float64 `json:"w"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

//A Kernel (convolution in Toeplitz form or dense) and A Bias
type Layer struct {
	Weight Kernel `json:"weight"`
	Bias   Bias   `json:"bias"`
}

type Stats struct {
	Iters    int
	Batch    int
	Corrects int
	Accuracy float64
	Time     int64
}

func NewStats(batch int) Stats {
	return Stats{Batch: batch}
}

func (s *Stats) Accumulate(other Stats) {
	s.Iters++
	s.Corrects += other.Corrects
	s.Accuracy += other.Accuracy
	s.Time += other.Time
}

func (s *Stats) PrintResult() {
	fmt.Println("---------------------------------------------------------------------------------")
	fmt.Println("[!] Results: ")
	fmt.Printf("Accuracy: %f\n", s.Accuracy/float64(s.Iters))
	fmt.Printf("Corrects / tot: %d / %d \n", s.Corrects, s.Iters*s.Batch)
	fmt.Printf("Avg Time for Eval: %f\n ms", float64(s.Time)/float64(s.Iters))
}

// Returns weight and bias of layer
func (l *Layer) Build(batchsize int) (*mat.Dense, *mat.Dense) {
	w := buildKernelMatrix(l.Weight)
	b := buildBiasMatrix(l.Bias, plainUtils.NumCols(w), batchsize)
	return w, b
}

func (l *Layer) BuildWeight() *mat.Dense {
	w := buildKernelMatrix(l.Weight)
	return w
}

func (l *Layer) BuildBias(batchsize int) *mat.Dense {
	w := buildKernelMatrix(l.Weight)
	b := buildBiasMatrix(l.Bias, plainUtils.NumCols(w), batchsize)
	return b
}

// Returns a matrix M s.t X x M = conv(x,layer), or a dense layer
func buildKernelMatrix(k Kernel) *mat.Dense {
	res := mat.NewDense(k.Rows, k.Cols, nil)
	for i := 0; i < k.Rows; i++ {
		for j := 0; j < k.Cols; j++ {
			res.Set(i, j, k.W[i*k.Cols+j])
		}
	}
	return res
}

// Compute a matrix containing the bias of the layer, to be added to the result of a Kernel multiplication
func buildBiasMatrix(b Bias, cols, batchSize int) *mat.Dense {
	res := mat.NewDense(batchSize, cols, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, plainUtils.Pad(b.B, cols-len(b.B)))
	}
	return res
}

//Returns number of correct values, accuracy and predicted values
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
