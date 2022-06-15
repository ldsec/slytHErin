package multidim

import (
	"fmt"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	"github.com/tuneinsight/lattigo/v3/utils"
	"math"

	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
)

type PackedMatrix struct {
	rows, cols, dim, n int
	M                  [][]*Matrix
}

func (pm *PackedMatrix) Rows() int {
	return pm.rows
}

func (pm *PackedMatrix) Cols() int {
	return pm.cols
}

func (pm *PackedMatrix) Dim() int {
	return pm.dim
}

// Pack matrix with padding
func PackMatrixSingle(m *mat.Dense, dim int) (pm *PackedMatrix) {
	raw_r, raw_c := m.Dims()
	pack_r := int(math.Ceil(float64(raw_r) / float64(dim)))
	pack_c := int(math.Ceil(float64(raw_c) / float64(dim)))
	pm = new(PackedMatrix)
	pm.rows, pm.cols, pm.dim, pm.n = pack_r, pack_c, dim, 1
	pm.M = make([][]*Matrix, pack_r*pack_c)
	for i := 0; i < pack_r; i++ {
		for j := 0; j < pack_c; j++ {
			st_i := i * dim
			st_j := j * dim
			en_i := plainUtils.Min((i+1)*dim, raw_r)
			en_j := plainUtils.Min((j+1)*dim, raw_c)
			m_slice := m.Slice(st_i, en_i, st_j, en_j).(*mat.Dense)
			pm.M[i*pack_c+j] = make([]*Matrix, 1)
			pm.M[i*pack_c+j][0] = NewMatrix(dim, dim)
			pm.M[i*pack_c+j][0].SetReal(dim, dim, m_slice)
		}
	}
	return pm
}

// Pack matrix with padding
func PackMatrixParallel(m *mat.Dense, dim, logSlots int) (pm *PackedMatrix) {
	raw_r, raw_c := m.Dims()
	pm = new(PackedMatrix)
	pm.n = plainUtils.Min((1<<logSlots)/(dim*dim), m.RawMatrix().Rows/dim)
	pack_r := int(math.Ceil(float64(raw_r) / float64(pm.n*dim)))
	pack_c := int(math.Ceil(float64(raw_c) / float64(dim)))
	pm.rows, pm.cols, pm.dim = pack_r, pack_c, dim
	pm.M = make([][]*Matrix, pack_r*pack_c)
	for i := 0; i < pack_r; i++ {
		for j := 0; j < pack_c; j++ {
			pm.M[i*pack_c+j] = make([]*Matrix, pm.n)
			st_j := j * dim
			en_j := plainUtils.Min((j+1)*dim, raw_c)
			for k := 0; k < pm.n; k++ {
				st_i := (i*pm.n + k) * dim
				en_i := plainUtils.Min(st_i+dim, raw_r)
				m_slice := m.Slice(st_i, en_i, st_j, en_j).(*mat.Dense)
				pm.M[i*pack_c+j][k] = NewMatrix(dim, dim)
				pm.M[i*pack_c+j][k].SetReal(dim, dim, m_slice)
			}
		}
	}
	return pm
}

// Pack matrix with padding
func PackAndTransposeParallel(m *mat.Dense, dim, logSlots int) (pm *PackedMatrix) {
	raw_r, raw_c := m.Dims()
	pm = new(PackedMatrix)
	pm.n = plainUtils.Min((1<<logSlots)/(dim*dim), m.RawMatrix().Rows/dim)
	pack_c := int(math.Ceil(float64(raw_r) / float64(pm.n*dim)))
	pack_r := int(math.Ceil(float64(raw_c) / float64(dim)))
	pm.rows, pm.cols, pm.dim = pack_r, pack_c, dim
	pm.M = make([][]*Matrix, pack_r*pack_c)
	for i := 0; i < pack_r; i++ {
		for j := 0; j < pack_c; j++ {
			pm.M[i*pack_c+j] = make([]*Matrix, pm.n)
			for k := 0; k < pm.n; k++ {
				st_j := i * dim
				en_j := plainUtils.Min((j+1)*dim, raw_c)
				st_i := (j*pm.n + k) * dim
				en_i := plainUtils.Min(st_i+dim, raw_r)
				m_slice := ExplicitTranspose(m.Slice(st_i, en_i, st_j, en_j))
				pm.M[i*pack_c+j][k] = NewMatrix(dim, dim)
				pm.M[i*pack_c+j][k].SetReal(dim, dim, m_slice)
			}
		}
	}
	return pm
}

// Pack matrix with padding
func PackMatrixParallelReplicated(m *mat.Dense, dim, n int) (pm *PackedMatrix) {
	pm = new(PackedMatrix)
	raw_r, raw_c := m.Dims()
	pack_r := int(math.Ceil(float64(raw_r) / float64(dim)))
	pack_c := int(math.Ceil(float64(raw_c) / float64(dim)))
	pm.rows, pm.cols, pm.dim, pm.n = pack_r, pack_c, dim, n
	pm.M = make([][]*Matrix, pack_r*pack_c)
	for i := 0; i < pack_r; i++ {
		for j := 0; j < pack_c; j++ {
			st_i := i * dim
			st_j := j * dim
			en_i := plainUtils.Min((i+1)*dim, raw_r)
			en_j := plainUtils.Min((j+1)*dim, raw_c)
			m_slice := m.Slice(st_i, en_i, st_j, en_j).(*mat.Dense)
			pm.M[i*pack_c+j] = make([]*Matrix, pm.n)
			for k := 0; k < pm.n; k++ {
				pm.M[i*pack_c+j][k] = NewMatrix(dim, dim)
				pm.M[i*pack_c+j][k].SetReal(dim, dim, m_slice)
			}
		}
	}
	return pm
}

// UnPack matrix with padding
func UnpackMatrixParallel(pm *PackedMatrix, dim, rows, cols int) (ret []float64) {
	ret = make([]float64, rows*cols)
	for i := 0; i < pm.rows; i++ {
		for j := 0; j < pm.cols; j++ {
			if j*dim >= cols {
				break
			}
			for k := 0; k < pm.n; k++ {
				if (i*pm.n+k)*dim >= rows {
					break
				}
				for ii := 0; ii < dim; ii++ {
					row := (i*pm.n+k)*dim + ii
					if row >= rows {
						break
					}
					for jj := 0; jj < dim; jj++ {
						col := j*dim + jj
						if col >= cols {
							break
						}
						ret[row*cols+col] = real(pm.M[i*pm.cols+j][k].M[ii*dim+jj])
					}
				}
			}
		}
	}
	return ret
}

// UnPack matrix with padding
func UnpackMatrixSingle(pm *PackedMatrix, dim, rows, cols int) (ret []float64) {
	ret = make([]float64, rows*cols)
	for i := 0; i < pm.rows; i++ {
		if i*dim >= rows {
			break
		}
		for j := 0; j < pm.cols; j++ {
			if j*dim >= cols {
				break
			}
			for ii := 0; ii < dim; ii++ {
				row := i*dim + ii
				if row >= rows {
					break
				}
				for jj := 0; jj < dim; jj++ {
					col := j*dim + jj
					if col >= cols {
						break
					}
					ret[row*cols+col] = real(pm.M[i*pm.cols+j][0].M[ii*dim+jj])
				}
			}
		}
	}
	return ret
}

func UnpackCipherParallel(ct *CiphertextBatchMatrix, dim, rows, cols int, e ckks2.Encoder, d ckks2.Decryptor, params ckks2.Parameters, n int) (ret []float64) {
	ret = make([]float64, rows*cols)
	if n == 0 {
		n = (1 << params.LogSlots()) / (dim * dim)
	}
	for i := 0; i < ct.rows; i++ {
		for j := 0; j < ct.cols; j++ {
			if j*dim >= cols {
				break
			}
			mat := e.Decode(d.DecryptNew(ct.M[i*ct.cols+j]), params.LogSlots())
			for k := 0; k < n; k++ {
				if (i*n+k)*dim >= rows {
					break
				}
				for ii := 0; ii < dim; ii++ {
					row := (i*n+k)*dim + ii
					if row >= rows {
						break
					}
					for jj := 0; jj < dim; jj++ {
						col := j*dim + jj
						if col >= cols {
							break
						}
						ret[row*cols+col] = real(mat[k*dim*dim+ii*dim+jj])
					}
				}
			}

		}
	}
	return ret
}

// UnPack matrix with padding
func UnpackCipherSingle(ct *CiphertextBatchMatrix, dim, rows, cols int, e ckks2.Encoder, d ckks2.Decryptor, params ckks2.Parameters) (ret []float64) {
	ret = make([]float64, rows*cols)
	for i := 0; i < ct.rows; i++ {
		if i*dim >= rows {
			break
		}
		for j := 0; j < ct.cols; j++ {
			if j*dim >= cols {
				break
			}
			mat := e.Decode(d.DecryptNew(ct.M[i*ct.cols+j]), params.LogSlots())
			//
			for ii := 0; ii < dim; ii++ {
				row := i*dim + ii
				if row >= rows {
					break
				}
				for jj := 0; jj < dim; jj++ {
					col := j*dim + jj
					if col >= cols {
						break
					}
					ret[row*cols+col] = real(mat[ii*dim+jj])
				}
			}
		}
	}
	return ret
}

// UnPack matrix with padding
func UnpackPlainParallel(pt *PlaintextBatchMatrix, dim, rows, cols int, e ckks2.Encoder, params ckks2.Parameters, n int) (ret []float64) {
	ret = make([]float64, rows*cols)
	if n == 0 {
		n = (1 << params.LogSlots()) / (dim * dim)
	}
	for i := 0; i < pt.rows; i++ {
		for j := 0; j < pt.cols; j++ {
			if j*dim >= cols {
				break
			}
			for k := 0; k < n; k++ {
				if (i*n+k)*dim >= rows {
					break
				}
				mat := e.Decode(pt.M[i*pt.cols+j][k], params.LogSlots())
				for ii := 0; ii < dim; ii++ {
					row := (i*n+k)*dim + ii
					if row >= rows {
						break
					}
					for jj := 0; jj < dim; jj++ {
						col := j*dim + jj
						if col >= cols {
							break
						}
						ret[row*cols+col] = real(mat[ii*dim+jj])
					}
				}
			}
		}
	}
	return ret
}

// UnPack matrix with padding
func UnpackPlainSingle(pt *PlaintextBatchMatrix, dim, rows, cols int, e ckks2.Encoder, params ckks2.Parameters) (ret []float64) {
	ret = make([]float64, rows*cols)
	for i := 0; i < pt.rows; i++ {
		if i*dim >= rows {
			break
		}
		for j := 0; j < pt.cols; j++ {
			if j*dim >= cols {
				break
			}
			mat := e.Decode(pt.M[i*pt.cols+j][0], params.LogSlots())
			//
			for ii := 0; ii < dim; ii++ {
				row := i*dim + ii
				if row >= rows {
					break
				}
				for jj := 0; jj < dim; jj++ {
					col := j*dim + jj
					if col >= cols {
						break
					}
					ret[row*cols+col] = real(mat[ii*dim+jj])
				}
			}
		}
	}
	return ret
}

// dims : dimension of the matrix elements
// rows : number of rows of matrix elements
// cols : number of cols of matrix elements
// n    : number of parallel matrices
func GenRandomRealPackedMatrices(dim, rows, cols, n int) (ppm *PackedMatrix) {
	ppm = new(PackedMatrix)
	ppm.rows, ppm.cols, ppm.dim, ppm.n = rows, cols, dim, n
	ppm.M = make([][]*Matrix, rows*cols)
	for i := range ppm.M {
		ppm.M[i] = GenRandomRealMatrices(dim, dim, n)
	}
	return
}

// dims : dimension of the matrix elements
// rows : number of rows of matrix elements
// cols : number of cols of matrix elements
// n    : number of parallel matrices
func GenRandomComplexPackedMatrices(dim, rows, cols, n int) (ppm *PackedMatrix) {
	ppm = new(PackedMatrix)
	ppm.rows, ppm.cols, ppm.dim, ppm.n = rows, cols, dim, n
	ppm.M = make([][]*Matrix, rows*cols)
	for i := range ppm.M {
		ppm.M[i] = GenRandomComplexMatrices(dim, dim, n)
	}
	return
}
func (m *PackedMatrix) Apply(A *PackedMatrix, f func(v float64) float64) {
	rowsA := A.rows
	colsA := A.cols
	res := make([][]*Matrix, rowsA*colsA)
	for x := 0; x < A.n; x++ {
		for i := 0; i < rowsA; i++ {
			for j := 0; j < colsA; j++ {
				if res[i*colsA+j] == nil {
					res[i*colsA+j] = make([]*Matrix, A.n)
				}

				if res[i*colsA+j][x] == nil {
					res[i*colsA+j][x] = NewMatrix(A.dim, A.dim)
				}
				res[i*colsA+j][x].Apply(A.M[i*colsA+j][x], f)
			}
		}
	}
	m.rows = A.rows
	m.cols = A.cols
	m.n = A.n
	m.M = res
}

func (m *PackedMatrix) Add(A, B *PackedMatrix) {
	rowsA := A.rows
	colsA := A.cols
	colsB := B.cols
	rowsB := B.rows

	if rowsA != rowsB || colsA != colsB {
		panic("input matrices are not compatible for addition")
	}

	if A.dim != B.dim {
		panic("input matrices do not have the same decomposition dimension")
	}

	if A.n != B.n {
		panic("input matrices do not have the same number of parallel matrices")
	}
	res := make([][]*Matrix, rowsA*colsB)
	for x := 0; x < A.n; x++ {
		for i := 0; i < A.rows; i++ {
			for j := 0; j < A.cols; j++ {
				if res[i*colsB+j] == nil {
					res[i*colsB+j] = make([]*Matrix, A.n)
				}

				if res[i*colsB+j][x] == nil {
					res[i*colsB+j][x] = NewMatrix(A.dim, A.dim)
				}
				res[i*colsB+j][x].Add(A.M[i*colsA+j][x], B.M[i*colsB+j][x])
			}
		}
	}
	m.rows = A.rows
	m.cols = B.cols
	m.n = A.n
	m.M = res
}

func (m *PackedMatrix) EleMul(A, B *PackedMatrix) {
	rowsA := A.rows
	colsA := A.cols
	colsB := B.cols
	rowsB := B.rows

	if rowsA != rowsB || colsA != colsB {
		panic("input matrices are not compatible for addition")
	}

	if A.dim != B.dim {
		panic("input matrices do not have the same decomposition dimension")
	}

	if A.n != B.n {
		panic("input matrices do not have the same number of parallel matrices")
	}
	res := make([][]*Matrix, rowsA*colsB)
	for x := 0; x < A.n; x++ {
		for i := 0; i < A.rows; i++ {
			for j := 0; j < A.cols; j++ {
				if res[i*colsB+j] == nil {
					res[i*colsB+j] = make([]*Matrix, A.n)
				}

				if res[i*colsB+j][x] == nil {
					res[i*colsB+j][x] = NewMatrix(A.dim, A.dim)
				}
				res[i*colsB+j][x].EleMul(A.M[i*colsA+j][x], B.M[i*colsB+j][x])
			}
		}
	}
	m.rows = A.rows
	m.cols = B.cols
	m.n = A.n
	m.M = res
}

func (m *PackedMatrix) Mul(A, B *PackedMatrix) {

	rowsA := A.rows
	colsA := A.cols
	colsB := B.cols
	rowsB := B.rows

	if colsA != rowsB {
		panic("input matrices are not compatible for multiplication")
	}

	if A.dim != B.dim {
		panic("input matrices do not have the same decomposition dimension")
	}

	if A.n != B.n {
		panic("input matrices do not have the same number of parallel matrices")
	}

	res := make([][]*Matrix, rowsA*colsB)
	tmp := new(Matrix)

	// Iterates over each parallel batch
	for x := 0; x < A.n; x++ {
		// Matrix mul of each parallel batch
		for i := 0; i < A.rows; i++ {
			for j := 0; j < colsB; j++ {
				for k := 0; k < colsA; k++ {
					tmp.Mul(A.M[i*colsA+k][x], B.M[j+k*colsB][x])

					if res[i*colsB+j] == nil {
						res[i*colsB+j] = make([]*Matrix, A.n)
					}

					if res[i*colsB+j][x] == nil {
						res[i*colsB+j][x] = NewMatrix(A.dim, A.dim)
					}

					res[i*colsB+j][x].Add(res[i*colsB+j][x], tmp)
				}
			}
		}
	}

	m.rows = A.rows
	m.cols = B.cols
	m.n = A.n
	m.dim = A.dim
	m.M = res
}

func (m *PackedMatrix) Transpose(A *PackedMatrix) {
	m.M = make([][]*Matrix, len(A.M))
	for i := range m.M {
		m.M[i] = make([]*Matrix, A.n)
	}
	tmp := make([]*Matrix, A.Cols()*A.Rows())
	for x := 0; x < A.n; x++ {

		for i := 0; i < len(A.M); i++ {
			m.M[i][x] = A.M[i][x].Transpose()
		}

		cols, rows := A.cols, A.rows

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				tmp[j*rows+i] = m.M[i*cols+j][x]
			}
		}

		for i := range tmp {
			m.M[i][x] = tmp[i]
		}
	}

	m.rows, m.cols, m.n, m.dim = A.cols, A.rows, A.n, A.dim
}

func (m *PackedMatrix) Print(batch int) {
	for i := range m.M {
		m.M[i][batch].Print()
	}
}

// Matrix is a struct holding a row flatened complex matrix.
type Matrix struct {
	rows, cols int
	Real       bool
	M          []complex128
}

// NewMatrix creates a new matrix.
func NewMatrix(rows, cols int) (m *Matrix) {
	m = new(Matrix)
	m.M = make([]complex128, rows*cols)
	m.rows = rows
	m.cols = cols
	m.Real = true
	return
}

func (m *Matrix) Copy() (mCopy *Matrix) {
	mCopy = new(Matrix)
	mCopy.M = make([]complex128, len(m.M))
	copy(mCopy.M, m.M)
	mCopy.rows = m.rows
	mCopy.cols = m.cols
	mCopy.Real = m.Real
	return
}

func (m *Matrix) Set(rows, cols int, v []complex128) {
	m.M = make([]complex128, len(v))
	copy(m.M, v)
	m.rows = rows
	m.cols = cols
}

func (m *Matrix) SetReal(rows, cols int, mm *mat.Dense) {
	m.M = make([]complex128, rows*cols)
	rc, cc := mm.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i >= rc || j >= cc {
				m.M[i*cols+j] = 0
			} else {
				m.M[i*cols+j] = complex(mm.At(i, j), 0)
			}
		}
	}
	m.rows = rows
	m.cols = cols
	m.Real = true
}

func (m *Matrix) SetRow(idx int, row []complex128) {
	for i := range row {
		m.M[i+idx*m.cols] = row[i]
	}
}

// Rows returns the number of rows of the matrix.
func (m *Matrix) Rows() int {
	return m.rows
}

// Cols returns the number of columns of the matrix.
func (m *Matrix) Cols() int {
	return m.cols
}

// Add adds matrix A and B and stores the result on the target.
func (m *Matrix) Add(A, B *Matrix) {

	if len(A.M) != len(B.M) {
		panic("input matrices are incompatible for addition")
	}

	if m.M == nil {
		m.M = make([]complex128, len(A.M))
	} else if len(m.M) > len(A.M) {
		m.M = m.M[:len(A.M)]
	} else if len(m.M) < len(A.M) {
		m.M = append(m.M, make([]complex128, len(A.M)-len(m.M))...)
	}

	for i := range A.M {
		m.M[i] = A.M[i] + B.M[i]
	}

	if m != A && m != B {
		m.Real = A.Real && B.Real
		m.rows = A.rows
		m.cols = A.cols
	} else if m != B {
		m.rows = B.rows
		m.cols = B.cols
		m.cols = B.cols
	}
}

// Add adds matrix A and B and stores the result on the target.
func (m *Matrix) EleMul(A, B *Matrix) {

	if len(A.M) != len(B.M) {
		panic("input matrices are incompatible for addition")
	}

	if m.M == nil {
		m.M = make([]complex128, len(A.M))
	} else if len(m.M) > len(A.M) {
		m.M = m.M[:len(A.M)]
	} else if len(m.M) < len(A.M) {
		m.M = append(m.M, make([]complex128, len(A.M)-len(m.M))...)
	}

	for i := range A.M {
		m.M[i] = A.M[i] * B.M[i]
	}

	if m != A && m != B {
		m.Real = A.Real && B.Real
		m.rows = A.rows
		m.cols = A.cols
	} else if m != B {
		m.rows = B.rows
		m.cols = B.cols
		m.cols = B.cols
	}
}

func (m *Matrix) Apply(A *Matrix, f func(v float64) float64) {

	if m.M == nil {
		m.M = make([]complex128, len(A.M))
	} else if len(m.M) > len(A.M) {
		m.M = m.M[:len(A.M)]
	} else if len(m.M) < len(A.M) {
		m.M = append(m.M, make([]complex128, len(A.M)-len(m.M))...)
	}

	for i := range A.M {
		m.M[i] = complex(f(real(A.M[i])), 0)
	}

	if m != A {
		m.Real = true
		m.rows = A.rows
		m.cols = A.cols
	}
}

func (m *Matrix) Abs() {
	for i := range m.M {
		m.M[i] = complex(math.Abs(real(m.M[i])), math.Abs(imag(m.M[i])))
	}
}

func (m *Matrix) Sub(A, B *Matrix) {

	if len(A.M) != len(B.M) {
		panic("input matrices are incompatible for addition")
	}

	if m.M == nil {
		m.M = make([]complex128, len(A.M))
	} else if len(m.M) > len(A.M) {
		m.M = m.M[:len(A.M)]
	} else if len(m.M) < len(A.M) {
		m.M = append(m.M, make([]complex128, len(A.M)-len(m.M))...)
	}

	for i := range A.M {
		m.M[i] = A.M[i] - B.M[i]
	}

	if m != A && m != B {
		m.Real = A.Real && B.Real
		m.rows = A.rows
		m.cols = A.cols
	} else if m != B {
		m.rows = B.rows
		m.cols = B.cols
		m.cols = B.cols
	}
}

func (m *Matrix) SumColumns(A *Matrix) {

	rowsA := A.Rows()
	colsA := A.Cols()

	acc := make([]complex128, colsA)

	for i := 0; i < colsA; i++ {
		for j := 0; j < rowsA; j++ {
			acc[i] += A.M[i+j*colsA]
		}
	}

	m.M = acc
	m.rows = 1
	m.cols = colsA
	m.Real = A.Real
}

func (m *Matrix) SumRows(A *Matrix) {

	rowsA := A.Rows()
	colsA := A.Cols()

	acc := make([]complex128, rowsA)

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			acc[i] += A.M[i*rowsA+j]
		}
	}

	m.M = acc
	m.rows = rowsA
	m.cols = 1
	m.Real = A.Real
}

func (m *Matrix) Dot(A, B *Matrix) {
	if A.Rows() != B.Rows() || A.Cols() != B.Cols() {
		panic("matrices are incompatible for dot product")
	}

	rowsA := A.Rows()
	colsA := A.Cols()

	acc := make([]complex128, rowsA*colsA)

	for i := range A.M {
		acc[i] = A.M[i] * B.M[i]
	}

	m.M = acc
	m.rows = rowsA
	m.cols = colsA
	m.Real = A.Real && B.Real
}

func (m *Matrix) Func(A *Matrix, f func(x complex128) complex128) {
	acc := make([]complex128, len(A.M))
	for i := range A.M {
		acc[i] = f(A.M[i])
	}
	m.M = acc
	m.rows = A.Rows()
	m.cols = A.Cols()
	m.Real = A.Real
}

func (m *Matrix) MultConst(A *Matrix, c complex128) {
	acc := make([]complex128, len(A.M))
	for i := range A.M {
		acc[i] = c * A.M[i]
	}
	m.M = acc
	m.rows = A.Rows()
	m.cols = A.Cols()
	m.Real = A.Real
}

// MulMat multiplies A with B and returns the result on the target.
func (m *Matrix) Mul(A, B *Matrix) {

	if A.Cols() != B.Rows() {
		panic("matrices are incompatible for multiplication")
	}

	rowsA := A.Rows()
	colsA := A.Cols()
	colsB := B.Cols()

	acc := make([]complex128, rowsA*colsB)

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				acc[i*colsB+j] += A.M[i*colsA+k] * B.M[j+k*colsB]
			}
		}
	}

	m.M = acc
	m.rows = A.Rows()
	m.cols = B.Cols()

	m.Real = A.Real && B.Real
}

// GenRandomComplexMatrices generates a list of complex matrices.
func GenRandomComplexMatrices(rows, cols, n int) (Matrices []*Matrix) {

	Matrices = make([]*Matrix, n)

	for k := range Matrices {
		m := NewMatrix(rows, cols)
		for i := 0; i < rows*cols; i++ {
			m.M[i] = complex(utils.RandFloat64(-1, 1), utils.RandFloat64(-1, 1))
			m.Real = false
		}
		Matrices[k] = m
	}

	return
}

// GenRandomReaMatrices generates a list of real matrices.
func GenRandomRealMatrices(rows, cols, n int) (Matrices []*Matrix) {

	Matrices = make([]*Matrix, n)

	for k := range Matrices {
		m := NewMatrix(rows, cols)
		for i := 0; i < rows*cols; i++ {
			m.M[i] = complex(utils.RandFloat64(-1, 1), 0)
			m.Real = true
		}
		Matrices[k] = m
	}

	return
}

// GenRandomReaMatrices generates a list of real matrices.
func GenZeroMatrices(rows, cols, n int) (Matrices []*Matrix) {

	Matrices = make([]*Matrix, n)

	for k := range Matrices {
		m := NewMatrix(rows, cols)
		for i := 0; i < rows*cols; i++ {
			m.M[i] = complex(0, 0)
			m.Real = true
		}
		Matrices[k] = m
	}

	return
}

// PermuteRows rotates each row by k where k is the row index.
// Equivalent to Transpoe(PermuteCols(Transpose(M)))
func (m *Matrix) PermuteRows() {
	var index int
	tmp := make([]complex128, m.Cols())
	for i := 0; i < m.Rows(); i++ {
		index = i * m.Cols()
		for j := range tmp {
			tmp[j] = m.M[index+j]
		}

		tmp = append(tmp[i:], tmp[:i]...)

		for j, c := range tmp {
			m.M[index+j] = c
		}
	}
}

// PermuteCols rotates each column by k, where k is the column index.
// Equivalent to Transpoe(PermuteRows(Transpose(M)))
func (m *Matrix) PermuteCols() {
	tmp := make([]complex128, m.Rows())
	for i := 0; i < m.Cols(); i++ {
		for j := range tmp {
			tmp[j] = m.M[i+j*m.Cols()]
		}

		tmp = append(tmp[i:], tmp[:i]...)

		for j, c := range tmp {
			m.M[i+j*m.Cols()] = c
		}
	}
}

// RotateCols rotates each column by k position to the left.
func (m *Matrix) RotateCols(k int) {

	k %= m.Cols()
	var index int
	tmp := make([]complex128, m.Cols())
	for i := 0; i < m.Rows(); i++ {
		index = i * m.Cols()
		for j := range tmp {
			tmp[j] = m.M[index+j]
		}

		tmp = append(tmp[k:], tmp[:k]...)

		for j, c := range tmp {
			m.M[index+j] = c
		}
	}
}

// RotateRows rotates each row by k positions to the left.
func (m *Matrix) RotateRows(k int) {
	k %= m.Rows()
	m.M = append(m.M[k*m.Cols():], m.M[:k*m.Cols()]...)
}

// Transpose transposes the matrix.
func (m *Matrix) Transpose() (mT *Matrix) {
	rows := m.Rows()
	cols := m.Cols()
	mT = NewMatrix(cols, rows)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			mT.M[rows*j+i] = m.M[i*cols+j]
		}
	}
	return
}

// Print prints the target matrix.
func (m *Matrix) Print() {

	if m.Real {
		fmt.Printf("[\n")
		for i := 0; i < m.Rows(); i++ {
			fmt.Printf("[ ")
			for j := 0; j < m.Cols(); j++ {
				fmt.Printf("%7.4f, ", real(m.M[i*m.Cols()+j]))
			}
			fmt.Printf("],\n")
		}
		fmt.Printf("]\n")
	} else {
		fmt.Printf("[")
		for i := 0; i < m.Rows(); i++ {
			fmt.Printf("[ ")
			for j := 0; j < m.Cols(); j++ {
				fmt.Printf("%7.4f, ", m.M[i*m.Cols()+j])
			}
			fmt.Printf("]\n")
		}
		fmt.Printf("]\n")
	}
}
