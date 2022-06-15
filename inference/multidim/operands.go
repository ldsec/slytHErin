package multidim

import (
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
)

type PlaintextBatchMatrix struct {
	rows, cols, dim int
	M               [][]*ckks2.Plaintext
}

func NewPlaintextBatchMatrix(rows, cols, dim int, pt [][]*ckks2.Plaintext) *PlaintextBatchMatrix {
	return &PlaintextBatchMatrix{rows, cols, dim, pt}
}

func (m *PlaintextBatchMatrix) Rows() int {
	return m.rows
}

func (m *PlaintextBatchMatrix) Cols() int {
	return m.cols
}

func (m *PlaintextBatchMatrix) Dim() int {
	return m.dim
}

func (m *PlaintextBatchMatrix) Level() int {
	return m.M[0][0].Level()
}

func (m PlaintextBatchMatrix) Scale() float64 {
	return m.M[0][0].Scale
}

type CiphertextBatchMatrix struct {
	rows, cols, dim int
	M               []*ckks2.Ciphertext
}

func NewCiphertextBatchMatrix(rows, cols, dim int, ct []*ckks2.Ciphertext) *CiphertextBatchMatrix {
	return &CiphertextBatchMatrix{rows, cols, dim, ct}
}

func AllocateCiphertextBatchMatrix(rows, cols, dim, level int, params ckks2.Parameters) (m *CiphertextBatchMatrix) {
	m = new(CiphertextBatchMatrix)
	m.rows = rows
	m.cols = cols
	m.dim = dim
	m.M = make([]*ckks2.Ciphertext, rows*cols)
	for i := range m.M {
		m.M[i] = ckks2.NewCiphertext(params, 1, level, 0)
	}
	return
}

func (m *CiphertextBatchMatrix) Rows() int {
	return m.rows
}

func (m *CiphertextBatchMatrix) Cols() int {
	return m.cols
}

func (m *CiphertextBatchMatrix) Dim() int {
	return m.dim
}

func (m *CiphertextBatchMatrix) Level() int {
	return m.M[0].Level()
}

func (m CiphertextBatchMatrix) Scale() float64 {
	return m.M[0].Scale
}
