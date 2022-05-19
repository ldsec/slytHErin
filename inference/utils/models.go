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

func InitReLU(deg int) PolyApprox {
	var relu PolyApprox
	if deg == 3 {
		relu.Degree = 3
		relu.Interval = 10.0
		relu.Coeffs = make([]float64, relu.Degree)
		relu.Coeffs[0] = 1.1155
		relu.Coeffs[1] = 5
		relu.Coeffs[2] = 4.4003
	} else if deg == 32 {
		relu.Degree = 32
		relu.Interval = 1.0
		MatLab := []float64{-1.0040897579718860e-53, 6.2085331754358028e-40, 9.4522902777573076e-50, -5.7963804324148821e-36, -4.0131279328625271e-46, 2.4410642683332394e-32, 1.0153477706512291e-42, -6.1290204181405624e-29, -1.7039434123075587e-39, 1.0216863193793685e-25, 1.9976235851829888e-36, -1.1917424918638167e-22, -1.6781853595392470e-33, 9.9891167268766684e-20, 1.0196230261578948e-30, -6.0833342283869143e-17, -4.4658877204790776e-28, 2.6909707871865122e-14, 1.3889468322950614e-25, -8.5600457797298628e-12, -2.9800845828620543e-23, 1.9200743786780711e-09, 4.2045289670858245e-21, -2.9487406547016763e-07, -3.6043867162675355e-19, 2.9886906932909647e-05, 1.6307741516672765e-17, -1.9601130409477464e-03, -2.8618809778714450e-16, 1.0678923596705732e-01, 5.0000000000000022e-01, 7.1225856852636027e-01}
		relu.Coeffs = make([]float64, len(MatLab))
		j := len(MatLab) - 1
		for i := 0; i < len(relu.Coeffs); i++ {
			relu.Coeffs[i] = MatLab[j-i]
			//fmt.Printf("%.4e * x^%d ", relu.Coeffs[i], i)
		}
	}
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

// Compute a matrix containing the bias of the layer, to be added to the result of a Kernel multiplication
func BuildBiasMatrix(b Bias, cols, batchSize int) *mat.Dense {
	res := mat.NewDense(batchSize, cols, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, plainUtils.Pad(b.B, cols-len(b.B)))
	}
	return res
}

// applies the activation function elementwise
func ActivatePlain(X *mat.Dense, activation PolyApprox) {
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

//computes how many levels are consumed by activation func
func (pa *PolyApprox) LevelsOfAct() int {
	return int(math.Ceil(math.Log2(float64(pa.Degree))))
}
