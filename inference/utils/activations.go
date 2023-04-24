// various utils, including methods for matrices in plaintext, model definitions and activation functions
package utils

import (
	"fmt"
	"github.com/ldsec/slytHErin/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"math"
)

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0.0
	}
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}
func SoftReLu(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}

func SiLU(x float64) float64 {
	return x * Sigmoid(x)
}

type PolyApprox interface {
	LevelsOfAct() int
	Rescale(w, b *mat.Dense) (*mat.Dense, *mat.Dense)
}

// Polynomial approximation from ckks Approximate
type ChebyPolyApprox struct {
	PolyApprox
	A, B      float64
	Degree    int
	Poly      *ckks.Polynomial
	ChebyBase bool
	F         func(x float64) float64
}

// Stores interval and degree of approximation in chebychev basis. Deg is set via SetDegOfParam
type ApproxParam struct {
	A   float64 `json:"a"`
	B   float64 `json:"b"`
	Deg int
}

// Approximation parameters for each layers
type ApproxParams struct {
	Params []ApproxParam `json:"intervals"`
}

// Initialize ReLU with coeffs not in cheby form from Matlab -> used by cryptonet
func InitReLU(deg int) *ChebyPolyApprox {
	relu := new(ChebyPolyApprox)
	relu.ChebyBase = false
	relu.A, relu.B = -10.0, 10.0
	if deg == 3 {
		relu.Degree = 3
		coeffs := make([]float64, relu.Degree)
		coeffs[0] = 1.1155
		coeffs[1] = 5
		coeffs[2] = 4.4003
		relu.Poly = ckks.NewPoly(plainUtils.RealToComplex(coeffs))
	} else if deg == 32 {
		relu.Degree = 32
		MatLab := []float64{-1.0040897579718860e-53, 6.2085331754358028e-40, 9.4522902777573076e-50, -5.7963804324148821e-36, -4.0131279328625271e-46, 2.4410642683332394e-32, 1.0153477706512291e-42, -6.1290204181405624e-29, -1.7039434123075587e-39, 1.0216863193793685e-25, 1.9976235851829888e-36, -1.1917424918638167e-22, -1.6781853595392470e-33, 9.9891167268766684e-20, 1.0196230261578948e-30, -6.0833342283869143e-17, -4.4658877204790776e-28, 2.6909707871865122e-14, 1.3889468322950614e-25, -8.5600457797298628e-12, -2.9800845828620543e-23, 1.9200743786780711e-09, 4.2045289670858245e-21, -2.9487406547016763e-07, -3.6043867162675355e-19, 2.9886906932909647e-05, 1.6307741516672765e-17, -1.9601130409477464e-03, -2.8618809778714450e-16, 1.0678923596705732e-01, 5.0000000000000022e-01, 7.1225856852636027e-01}
		coeffs := make([]float64, len(MatLab))
		j := len(MatLab) - 1
		for i := 0; i < len(coeffs); i++ {
			coeffs[i] = MatLab[j-i]
			//fmt.Printf("%.4e * x^%d ", relu.Coeffs[i], i)
		}
		relu.Poly = ckks.NewPoly(plainUtils.RealToComplex(coeffs))
	}
	return relu
}

// Initiliazes activation layer with function to approximate
func InitActivationCheby(act string, a, b float64, deg int) *ChebyPolyApprox {
	approx := new(ChebyPolyApprox)
	approx.A = a
	approx.B = b
	approx.Degree = deg
	var f func(x float64) float64
	if act == "soft relu" {
		f = SoftReLu
	} else if act == "silu" {
		f = SiLU
	}
	approx.Poly = ckks.Approximate(f, a, b, deg)
	approx.F = f
	approx.ChebyBase = true
	return approx
}

// applies the activation function elementwise. Needs rescaling first
func (activation *ChebyPolyApprox) ActivatePlain(X *mat.Dense) {
	rows, cols := X.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := X.At(r, c) //* (2.0 / (activation.B - activation.A))
			res := 0.0
			if activation.ChebyBase {
				//remove rescale for original f
				v = (X.At(r, c) - (-activation.A-activation.B)/(activation.B-activation.A))
				v = v / (2.0 / (activation.B - activation.A))
				res = activation.F(v)
			} else {
				for deg := 0; deg < activation.Degree; deg++ {
					res += (math.Pow(v, float64(deg)) * real(activation.Poly.Coeffs[deg]))
				}
			}
			X.Set(r, c, res)
		}
	}
}

// computes how many levels are consumed by activation func
func (approx *ChebyPolyApprox) LevelsOfAct() int {
	return approx.Poly.Depth()
}

// rescale weights for polynomial activation
func (approx *ChebyPolyApprox) Rescale(w, b *mat.Dense) (*mat.Dense, *mat.Dense) {
	mulC := 2.0 / (approx.B - approx.A)
	addC := (-approx.A - approx.B) / (approx.B - approx.A)

	wR := plainUtils.MulByConst(w, mulC)
	bR := plainUtils.AddConst(plainUtils.MulByConst(b, mulC), addC)
	return wR, bR
}

// decides the degree of approximation for each Param
func SetDegOfParam(Params ApproxParams) ApproxParams {
	ParamsNew := make([]ApproxParam, len(Params.Params))
	margin := 1.0
	for i, Param := range Params.Params {
		Param.A = math.Floor(Param.A) - margin
		Param.B = math.Floor(Param.B) + margin
		diff := Param.B - Param.A
		if diff <= 2 {
			Param.Deg = 3
		} else if diff <= 4 {
			Param.Deg = 7
		} else if diff <= 8 {
			Param.Deg = 15
		} else if diff <= 12 {
			Param.Deg = 31
		} else {
			Param.Deg = 63
		}
		fmt.Printf("Layer %d Approx: A = %f, B=%f --> deg = %d\n", i+1, Param.A, Param.B, Param.Deg)
		ParamsNew[i] = Param
	}
	return ApproxParams{ParamsNew}
}
