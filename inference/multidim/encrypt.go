package multidim

import (
	"fmt"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	ring2 "github.com/ldsec/lattigo/v2/ring"
	utils2 "github.com/ldsec/lattigo/v2/utils"
)

type BatchEncryptor struct {
	params    ckks2.Parameters
	encoder   ckks2.Encoder
	encryptor ckks2.Encryptor
	pool      *ring2.Poly
	values    []complex128
}

func NewBatchEncryptor(params ckks2.Parameters, key interface{}) (be *BatchEncryptor) {
	be = new(BatchEncryptor)
	be.params = params
	be.encoder = ckks2.NewEncoder(params)
	be.encryptor = ckks2.NewEncryptor(params, key)
	be.pool = params.RingQ().NewPoly()
	be.values = make([]complex128, params.Slots())
	return
}

func (be *BatchEncryptor) EncodeAndEncrypt(level int, scale float64, pm *PackedMatrix) (ct *CiphertextBatchMatrix) {
	logSlots := be.params.LogSlots()
	values := be.values

	pt := ckks2.NewPlaintextAtLevelFromPoly(level, be.pool)
	pt.Scale = scale

	w := make([]*ckks2.Ciphertext, pm.Rows()*pm.Cols())
	d := pm.M[0][0].Rows() * pm.M[0][0].Cols()
	nbMatrices := utils2.MinInt(len(pm.M[0]), (1<<logSlots)/d)

	for i := 0; i < pm.Rows()*pm.Cols(); i++ {
		for j := 0; j < nbMatrices; j++ {
			tmp := pm.M[i][j].M
			fmt.Printf("Matrix slot inside pm at coord %d, batch %d\n", i, j)
			fmt.Println(tmp)
			for k, c := range tmp {
				values[j*d+k] = c
			}
			fmt.Printf("finished putting in plaintext matrix in entry %d of batch %d\n", i, j)
			fmt.Println(values)
		}
		fmt.Printf("finished putting in plaintext all batches of entry %d\n", i)
		fmt.Println(values)

		be.encoder.EncodeNTT(pt, values, logSlots)
		w[i] = be.encryptor.EncryptNew(pt)
	}

	return NewCiphertextBatchMatrix(pm.Rows(), pm.Cols(), pm.Dim(), w)
}

func (be *BatchEncryptor) EncodeSingle(level int, scale float64, pm *PackedMatrix) *PlaintextBatchMatrix {
	w := make([][]*ckks2.Plaintext, pm.Rows()*pm.Cols())
	for i := 0; i < pm.Rows()*pm.Cols(); i++ {
		w[i] = []*ckks2.Plaintext{be.encoder.EncodeNTTNew(pm.M[i][0].M, be.params.LogSlots())}
	}
	return NewPlaintextBatchMatrix(pm.rows, pm.cols, pm.dim, w)
}

func (be *BatchEncryptor) EncodeParallel(level int, scale float64, pm *PackedMatrix) *PlaintextBatchMatrix {
	w := make([][]*ckks2.Plaintext, pm.Rows()*pm.Cols())
	values := be.values
	d := pm.M[0][0].Rows() * pm.M[0][0].Cols()
	for i := 0; i < pm.Rows()*pm.Cols(); i++ {
		for j := 0; j < pm.n; j++ {
			tmp := pm.M[i][j].M
			for k, c := range tmp {
				values[j*d+k] = c
			}
		}
		w[i] = []*ckks2.Plaintext{be.encoder.EncodeNTTNew(values, be.params.LogSlots())}
	}
	return NewPlaintextBatchMatrix(pm.rows, pm.cols, pm.dim, w)
}

// 	return NewCiphertextBatchMatrix(pm.Rows(), pm.Cols(), pm.Dim(), w)
// }
func (be *BatchEncryptor) EncodeForLeftMul(level int, pm *PackedMatrix) *PlaintextBatchMatrix {

	params := be.params
	encoder := be.encoder
	rows := pm.Rows()
	cols := pm.Cols()
	dim := pm.Dim()

	pt := make([][]*ckks2.Plaintext, rows*cols)
	for i := 0; i < rows*cols; i++ {

		// Diagonalized L0 encoding (plaintext)
		pt[i] = make([]*ckks2.Plaintext, dim)

		var values []complex128
		for j := 0; j < dim; j++ {

			values = make([]complex128, params.Slots())

			for k, matrix := range pm.M[i] {

				m := matrix.M
				// Each diagonal value
				for u := 0; u < dim; u++ {
					// i rotation index
					c := m[(u*dim)+(u+j)%dim]

					// replicates the value #dimension time
					for v := 0; v < dim; v++ {
						values[k*dim*dim+u*dim+v] = c
					}
				}
			}

			pt[i][j] = ckks2.NewPlaintext(params, level, params.QiFloat64(level))
			encoder.EncodeNTT(pt[i][j], values, params.LogSlots())
		}
	}

	return &PlaintextBatchMatrix{rows, cols, dim, pt}
}
