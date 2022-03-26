package plainUtils

import "gonum.org/v1/gonum/mat"

type Padding struct {
	//see func below
	P1       [][]float64
	P2       [][]float64
	inputDim int
	padding  int
}

//pads v with n 0s
func Pad(v []float64, n int) []float64 {
	res := make([]float64, len(v)+n)
	for i := range v {
		res[i] = v[i]
	}
	return res
}

/*
		Returns P1 and P2 matrixes such that:
			P2 * X * P1 = padded X, where * is the matrix product
		E.g is X = |a b|
	               |c d|
		if pad = 1, X becomes |0 0 0 0|
	                          |0 a b 0|
							  |0 c d 0|
	                          |0 0 0 0|
*/
func GenPaddingMatrixes(inputDim, padding int) *Padding {

	P1mat := mat.NewDense(inputDim, 2*padding+inputDim, nil)
	for i := 0; i < inputDim; i++ {
		j := i + padding
		P1mat.Set(i, j, 1.0)
	}
	P2mat := P1mat.T()

	P1 := make([][]float64, inputDim)
	for i := range P1 {
		P1[i] = make([]float64, 2*padding+inputDim)
		for j := range P1[i] {
			P1[i][j] = P1mat.At(i, j)
		}
	}
	P2 := make([][]float64, 2*padding+inputDim)
	for i := range P2 {
		P2[i] = make([]float64, inputDim)
		for j := range P2[i] {
			P2[i][j] = P2mat.At(i, j)
		}
	}
	P := new(Padding)
	P.P1 = P1
	P.P2 = P2
	P.padding = padding
	P.inputDim = inputDim
	return P
}
