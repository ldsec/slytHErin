package utils

import (
	"encoding/csv"
	"fmt"
	"github.com/ldsec/slytHErin/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
	"math"
	"os"
	"path/filepath"
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

// A Kernel (convolution in Toeplitz form or dense) and A Bias
type Layer struct {
	Weight Kernel `json:"weight"`
	Bias   Bias   `json:"bias"`
}

type Stats struct {
	Iters    int
	Batch    int
	Corrects []int
	Accuracy []float64
	Time     []int64 // Add slice to store time for each evaluation
}

func NewStats(batch int) Stats {
	return Stats{Batch: batch}
}

func (s *Stats) Accumulate(other Stats) {
	s.Iters++
	s.Corrects = append(s.Corrects, other.Corrects...)
	s.Accuracy = append(s.Accuracy, other.Accuracy...)
	s.Time = append(s.Time, other.Time...)
}

func (s *Stats) PrintResult(filePath string) error {
	fmt.Println("---------------------------------------------------------------------------------")
	if filePath != "" {
		// Check if the file exists, and create it if it doesn't
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			err := os.MkdirAll(filepath.Dir(filePath), os.ModePerm)
			if err != nil {
				return err
			}
			file, err := os.Create(filePath)
			if err != nil {
				return err
			}
			file.Close()
		}

		// Open the file for writing
		file, err := os.OpenFile(filePath, os.O_APPEND|os.O_WRONLY, os.ModeAppend)
		if err != nil {
			return err
		}
		defer file.Close()

		// Write the results to the file
		writer := csv.NewWriter(file)
		defer writer.Flush()

		// Write the header row
		header := []string{"Batch", "Total", "Accuracy", "Corrects", "Time", "StdDev"}
		writer.Write(header)

		// Write the data row
		totCorrects := 0
		for i := 0; i < len(s.Time); i++ {
			data := []string{
				fmt.Sprintf("%d", s.Batch),
				fmt.Sprintf("%d", s.Iters*s.Batch),
				fmt.Sprintf("%f", s.Accuracy[i]),
				fmt.Sprintf("%d", s.Corrects[i]),
				fmt.Sprintf("%d", s.Time[i]),
				fmt.Sprintf("%f", 0.0),
			}
			totCorrects += s.Corrects[i]
			writer.Write(data)
		}
		avgAcc := calculateAverage(toInterfaceSliceFloat64(s.Accuracy))
		avgTime, stdTime := calculateAverage(toInterfaceSliceInt64(s.Time)), calculateStdDev(toInterfaceSliceInt64(s.Time))

		data := []string{
			fmt.Sprintf("%d", s.Batch),
			fmt.Sprintf("%d", s.Iters*s.Batch),
			fmt.Sprintf("%f", avgAcc),
			fmt.Sprintf("%d", totCorrects),
			fmt.Sprintf("%f", avgTime),
			fmt.Sprintf("%f", stdTime),
		}
		writer.Write(data)
	}
	avgAcc := calculateAverage(toInterfaceSliceFloat64(s.Accuracy))
	avgTime, stdTime := calculateAverage(toInterfaceSliceInt64(s.Time)), calculateStdDev(toInterfaceSliceInt64(s.Time))
	totCorrects := 0
	for i := 0; i < len(s.Time); i++ {
		totCorrects += s.Corrects[i]
	}
	fmt.Println("[!] Results: ")
	fmt.Println("Accuracy:")
	fmt.Println(s.Accuracy)
	fmt.Printf("Avg Accuracy: %f\n", avgAcc)
	fmt.Printf("Corrects / tot: %d / %d\n", totCorrects, s.Iters*s.Batch)
	fmt.Printf("Avg Time for Eval: %f ms (+/- %f ms)\n", avgTime, stdTime)

	return nil
}

func toInterfaceSliceFloat64(slice []float64) []interface{} {
	interfaceSlice := make([]interface{}, len(slice))
	for i, v := range slice {
		interfaceSlice[i] = v
	}
	return interfaceSlice
}

func toInterfaceSliceInt64(slice []int64) []interface{} {
	interfaceSlice := make([]interface{}, len(slice))
	for i, v := range slice {
		interfaceSlice[i] = v
	}
	return interfaceSlice
}

func toInterfaceSliceInt(slice []int) []interface{} {
	interfaceSlice := make([]interface{}, len(slice))
	for i, v := range slice {
		interfaceSlice[i] = v
	}
	return interfaceSlice
}

func calculateAverage(values []interface{}) float64 {
	sum := 0.0
	for _, v := range values {
		switch value := v.(type) {
		case int:
			sum += float64(value)
		case int64:
			sum += float64(value)
		case float64:
			sum += value
		}
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []interface{}) float64 {
	n := len(values)
	if n < 2 {
		return 0.0
	}
	mean := calculateAverage(values)
	sum := 0.0
	for _, v := range values {
		switch value := v.(type) {
		case int:
			diff := float64(value) - mean
			sum += diff * diff
		case int64:
			diff := float64(value) - mean
			sum += diff * diff
		case float64:
			diff := value - mean
			sum += diff * diff
		}
	}
	variance := sum / float64(n-1)
	return math.Sqrt(variance)
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

// Returns number of correct values, accuracy and predicted values
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
