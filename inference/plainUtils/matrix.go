package plainUtils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

func NewDense(X [][]float64) *mat.Dense {
	return mat.NewDense(len(X), len(X[0]), Vectorize(X, true))
}

func PadDense(m *mat.Dense, rowPad, colPad int) *mat.Dense {
	n := mat.NewDense(NumRows(m)+rowPad, NumCols(m)+colPad, nil)
	for i := 0; i < NumRows(m); i++ {
		for j := 0; j < NumCols(m); j++ {
			n.Set(i, j, m.At(i, j))
		}
	}
	return n
}

func DenseToMatrix(m *mat.Dense) [][]float64 {
	M := make([][]float64, NumRows(m))
	for i := 0; i < NumRows(m); i++ {
		M[i] = make([]float64, NumCols(m))
		for j := 0; j < NumCols(m); j++ {
			M[i][j] = m.At(i, j)
		}
	}
	return M
}

func TransposeDense(m *mat.Dense) (mt *mat.Dense) {
	mt = mat.NewDense(NumCols(m), NumRows(m), nil)
	for i := 0; i < NumRows(m); i++ {
		for j := 0; j < NumCols(m); j++ {
			mt.Set(j, i, m.At(i, j))
		}
	}
	return
}

func RandMatrix(r, c int) *mat.Dense {
	rand.Seed(42)
	m := make([]float64, r*c)
	for i := range m {
		m[i] = rand.Float64()
	}
	return mat.NewDense(r, c, m)
}

//returns a matrix useful for debug. E.g if r,c = 3,3 -> returns
// | 1 2 3 |
// | 4 5 6 |
// | 7 8 9 |
func MatrixForDebug(r, c int) *mat.Dense {
	m := make([]float64, r*c)
	for i := range m {
		m[i] = float64(i) + 1.0
	}
	return mat.NewDense(r, c, m)
}

func MulByConst(m *mat.Dense, c float64) *mat.Dense {
	v := MatToArray(m)
	for i := range v {
		for j := range v[0] {
			v[i][j] *= c
		}
	}
	return mat.NewDense(NumRows(m), NumCols(m), Vectorize(v, true))
}

func AddConst(m *mat.Dense, c float64) *mat.Dense {
	v := MatToArray(m)
	for i := range v {
		for j := range v[0] {
			v[i][j] += c
		}
	}
	return mat.NewDense(NumRows(m), NumCols(m), Vectorize(v, true))
}

func MatToArray(m *mat.Dense) [][]float64 {
	v := make([][]float64, NumRows(m))
	for i := 0; i < NumRows(m); i++ {
		v[i] = mat.Row(nil, i, m)
	}
	return v
}

func RowFlatten(m *mat.Dense) []float64 {
	v := make([][]float64, NumRows(m))
	for i := 0; i < NumRows(m); i++ {
		v[i] = mat.Row(nil, i, m)
	}
	return Vectorize(v, true)
}

func NumRows(m *mat.Dense) int {
	rows, _ := m.Dims()
	return rows
}

func NumCols(m *mat.Dense) int {
	_, cols := m.Dims()
	return cols
}

//replicates v ,n times
func ReplicateValue(v float64, n int) []float64 {
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		res[i] = v
	}
	return res
}

// eye returns a new identity matrix of size n×n.
func Eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

//	Input: matrix X
//	Output: column array of vectorized X
//	Example:
//
//	X = |a b|
//		|c d|
//
//	if tranpose false:
//		output = [a ,b, c, d] column vector
//	else:
//		output = [a ,c, b, d] column vector
//
//	for reference: https://en.wikipedia.org/wiki/Vectorization_(mathematics)
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
	/*
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
	*/
	return X_flat
}

func Distance(a, b []float64) float64 {
	//computes euclidean distance between arrays
	d := 0.0
	for i := range a {
		d += math.Pow(a[i]-b[i], 2.0)
	}
	return math.Sqrt(d)
}

func ComplexToReal(v []complex128) []float64 {
	c := make([]float64, len(v))
	for i := range v {
		c[i] = real(v[i])
	}
	return c
}

func RealToComplex(v []float64) []complex128 {
	c := make([]complex128, len(v))
	for i := range v {
		c[i] = complex(v[i], 0.0)
	}
	return c
}

func MulByi(v []float64) []complex128 {
	vi := make([]complex128, len(v))
	for i := range vi {
		vi[i] = complex(0, v[i])
	}
	return vi
}

//rotates v of k positions to the left or to the right if k < 0
func RotateRealArray(v []float64, k int) []float64 {
	if k == 0 || len(v) == 0 {
		return v
	}
	var r int
	if k < 0 {
		//Right
		r = len(v) - k%len(v)
	} else {
		r = k % len(v)
	}
	v = append(v[r:], v[:r]...)
	return v
}

//rotates v of k positions to the left or to the right if k < 0
func RotateComplexArray(v []complex128, k int) []complex128 {
	if k == 0 || len(v) == 0 {
		return v
	}
	var r int
	if k < 0 {
		//Right
		r = len(v) - k%len(v)
	} else {
		r = k % len(v)
	}
	v = append(v[r:], v[:r]...)
	return v
}

//replicates v with n copies
func ReplicateRealArray(v []float64, n int) []float64 {
	vr := make([]float64, n*len(v))
	for i := range vr {
		vr[i] = v[i%len(v)]
	}
	return vr
}

//replicates v with n copies
func ReplicateComplexArray(v []complex128, n int) []complex128 {
	vr := make([]complex128, n*len(v))
	for i := range vr {
		vr[i] = v[i%len(v)]
	}
	return vr
}

func ApplyFuncDense(f func(v float64) float64, a *mat.Dense) *mat.Dense {
	m := mat.NewDense(NumRows(a), NumCols(a), nil)
	for i := 0; i < NumRows(a); i++ {
		for j := 0; j < NumCols(a); j++ {
			m.Set(i, j, f(a.At(i, j)))
		}
	}
	return m
}

func PrintDense(m *mat.Dense) {
	v := MatToArray(m)
	for _, r := range v {
		fmt.Printf("|")
		for _, c := range r {
			fmt.Printf("%f ", c)
		}
		fmt.Println("|")
	}
}
