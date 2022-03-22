package plainUtils

import "gonum.org/v1/gonum/mat"

// eye returns a new identity matrix of size n×n.
func eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

/*
	Input: matrix X
	Output: column array of vectorized X
	Example:

	X = |a b|
		|c d|

	if tranpose false:
		output = [a ,b, c, d] column vector
	else:
		output = [a ,c, b, d] column vector

	for reference: https://en.wikipedia.org/wiki/Vectorization_(mathematics)
*/
func Vectorize(X [][]float64, transpose bool) []float64 {
	rows := len(X)
	cols := len(X[0])
	X_flat := make([]float64, rows*cols)
	if !transpose {
		// tranpose X --> output will be flatten(X)
		for j := 0; j < cols; j++ {
			for i := 0; i < rows; i++ {
				X_flat[j*rows+i] = X[i][j]
			}
		}
	} else {
		// do not tranpose X --> output will be flatten(X.T)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				X_flat[i*cols+j] = X[i][j]
			}
		}
	}
	X_mat := mat.NewDense(rows, cols, X_flat)

	// create #cols canon-basis vectors. Each i-th vector has 1 in the i-th position
	c_bases := make([]*mat.Dense, cols)
	for i := 0; i < cols; i++ {
		c_bases[i] = mat.NewDense(cols, 1, nil)
		c_bases[i].Set(i, 0, 1)
	}
	// create the block matrixes
	B := make([]*mat.Dense, cols)
	for i := 0; i < cols; i++ {
		B[i] = mat.NewDense(rows*cols, rows, nil)
		I := eye(rows)
		B[i].Kronecker(c_bases[i], I)
	}
	X_vec := mat.NewDense(rows*cols, 1, nil)
	for i := 0; i < cols; i++ {
		tmpA := mat.NewDense(rows, 1, nil)
		tmpA.Mul(X_mat, c_bases[i])
		tmpB := mat.NewDense(rows*cols, 1, nil)
		tmpB.Mul(B[i], tmpA)
		X_vec.Add(X_vec, tmpB)
	}
	for i := 0; i < rows*cols; i++ {
		X_flat[i] = X_vec.At(i, 0)
	}
	return X_flat
}
