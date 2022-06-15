package multidim

import (
	"fmt"

	"math"

	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
)

type MatrixMultiplicationLiteral struct {
	Dimension   int
	LevelStart  int
	InputScale  float64
	TargetScale float64
}

// MatrixMultiplication is a struct storing the linear transformations necessary for the homomorphic
// multiplication of square matrices.
type MatrixMultiplication struct {
	dimension   int
	levelstart  int
	targetScale float64
	ckks2.PtDiagMatrix
	PermuteRows ckks2.PtDiagMatrix
	PermuteCols ckks2.PtDiagMatrix
	RotRows     []ckks2.PtDiagMatrix
	RotCols     []ckks2.PtDiagMatrix
}

func (mm *MatrixMultiplication) StringMap() string {
	return fmt.Sprintf("0x%04o0x%04o", mm.dimension, mm.levelstart)
}

func (mm *MatrixMultiplication) Rotations(params ckks2.Parameters) []int {
	rotations := []int{}
	rotations = append(rotations, params.RotationsForDiagMatrixMult(mm.PermuteRows)...)
	rotations = append(rotations, params.RotationsForDiagMatrixMult(mm.PermuteCols)...)
	for i := range mm.RotRows {
		rotations = append(rotations, params.RotationsForDiagMatrixMult(mm.RotRows[i])...)
	}
	for i := range mm.RotCols {
		rotations = append(rotations, params.RotationsForDiagMatrixMult(mm.RotCols[i])...)
	}
	return rotations
}

// GenMatMulLinTrans generates the plaintext linear transformation necessary for the homomorphic
// multiplication of square matrices.
func NewMatrixMultiplicatonFromLiteral(params ckks2.Parameters, mmParams MatrixMultiplicationLiteral, encoder ckks2.Encoder) (mm MatrixMultiplication) {
	scale := params.QiFloat64(mmParams.LevelStart) * math.Sqrt(params.QiFloat64(mmParams.LevelStart-2)/mmParams.InputScale)

	mm.dimension = mmParams.Dimension
	mm.levelstart = mmParams.LevelStart
	mm.targetScale = mmParams.TargetScale
	mm.PermuteRows = GenPermuteRowsMatrix(mmParams.LevelStart, scale, 4.0, mmParams.Dimension, params.LogSlots(), encoder)
	mm.PermuteCols = GenPermuteColsMatrix(mmParams.LevelStart, scale, 4.0, mmParams.Dimension, params.LogSlots(), encoder)

	mm.RotCols = make([]ckks2.PtDiagMatrix, mmParams.Dimension-1)
	mm.RotRows = make([]ckks2.PtDiagMatrix, mmParams.Dimension-1)

	for i := 0; i < mmParams.Dimension-1; i++ {
		mm.RotCols[i] = GenSubVectorRotationMatrix(mmParams.LevelStart, params.QiFloat64(mmParams.LevelStart-1)*(mmParams.TargetScale/mmParams.InputScale), mmParams.Dimension, i+1, params.LogSlots(), encoder)
		mm.RotRows[i] = GenSubVectorRotationMatrix(mmParams.LevelStart, params.QiFloat64(mmParams.LevelStart-1)*(mmParams.TargetScale/mmParams.InputScale), mmParams.Dimension*mmParams.Dimension, (i+1)*mmParams.Dimension, params.LogSlots(), encoder)
	}
	return
}

// GenPermuteRowsMatrix rotates each row of the matrix by k position, where k is the row index.
func GenPermuteRowsMatrix(level int, scale, maxM1N2Ratio float64, dimension int, logSlots int, encoder ckks2.Encoder) ckks2.PtDiagMatrix {

	slots := 1 << logSlots

	diagMatrix := make(map[int][]complex128)

	d2 := int(dimension * dimension)

	for i := -int(dimension) + 1; i < int(dimension); i++ {

		m := make([]complex128, slots)

		for k := 0; k < d2; k++ {

			if i < 0 {
				for j := i; j < int(dimension); j++ {
					x := (d2 + k - (int(dimension)+i)*int(dimension)) % d2
					if x < int(dimension) && x >= -i {
						m[k] = 1
					}
				}
			} else {

				for j := i; j < int(dimension); j++ {
					if (d2+k-int(dimension)*i)%d2 < int(dimension)-i {
						m[k] = 1
					}
				}
			}
		}

		populateVector(m, d2, logSlots)

		diagMatrix[(i+int(slots))%slots] = m
	}

	return encoder.EncodeDiagMatrixBSGSAtLvl(level, diagMatrix, scale, maxM1N2Ratio, logSlots)

}

// GenPermuteColsMatrix rotates each column of the matrix by k position, where k is the column index.
func GenPermuteColsMatrix(level int, scale, maxM1N2Ratio float64, dimension int, logSlots int, encoder ckks2.Encoder) ckks2.PtDiagMatrix {

	slots := 1 << logSlots

	diagMatrix := make(map[int][]complex128)

	d2 := int(dimension * dimension)

	if d2 < slots {

		for i := -int((dimension - 1) * dimension); i < d2; i = i + int(dimension) {

			m := make([]complex128, 1<<logSlots)

			if i >= 0 {
				for j := 0; j < d2-i; j = j + int(dimension) {
					m[i/int(dimension)+j] = 1
				}
			} else {
				for j := 0; j < d2+i; j = j + int(dimension) {
					m[-i+int(dimension)+(i/int(dimension))+j] = 1
				}
			}

			populateVector(m, d2, logSlots)

			diagMatrix[(i+int(slots))%slots] = m

		}
	} else {
		for i := 0; i < int(dimension); i++ {

			m := make([]complex128, 1<<logSlots)

			for j := 0; j < d2; j = j + int(dimension) {
				m[j+i] = 1
			}

			populateVector(m, d2, logSlots)

			diagMatrix[i*int(dimension)] = m
		}
	}

	return encoder.EncodeDiagMatrixBSGSAtLvl(level, diagMatrix, scale, maxM1N2Ratio, logSlots)
}

// GenSubVectorRotationMatrix allows to generate a permutation matrix that roates subvectors independently.
// Given a vector of size N=2^"logSlots", partitionned into N/"vectorSize" subvectors each of size "vectorSize",
// rotates each subvector by "k" positions to the left.
//
// Example :
// Given v = [a_(0), a_(1), a_(2), ..., a_(N-3), a_(N-2), a_(N-1)],
// Then M x v = [rotate(a_(0), a_(1), ..., a_(vectorsize-1), k), ... , rotate(a_(N-vectorsize-1), a_(N-vectorsize), ..., a_(N-1), k)]
//
// If vectorSize does not divide N, then the last N%vectorSize slots are zero.
// If N = vectorSize, then no mask is generated and the evaluation is instead a single rotation.
//
// This is done by generating the two masks :
//       	 |     vectorsize     |, ..., |     vectorsize     |
// mask_0 = [{1, ..., 1, 0, ..., 0}, ..., {1, ..., 1, 0, ..., 0}]
// mask_1 = [{0, ..., 0, 1, ..., 1}, ..., {0, ..., 0, 1, ..., 1}]
//            0 ----- k                    0 ----- k
func GenSubVectorRotationMatrix(level int, scale float64, vectorSize, k int, logSlots int, encoder ckks2.Encoder) (matrix ckks2.PtDiagMatrix) {

	k %= vectorSize

	diagMatrix := make(map[int][]complex128)

	slots := 1 << logSlots

	matrix.Vec = make(map[int]rlwe2.PolyQP)

	if vectorSize < slots {
		m0 := make([]complex128, slots)
		m1 := make([]complex128, slots)

		for i := 0; i < slots/vectorSize; i++ {

			index := i * vectorSize

			for j := 0; j < k; j++ {
				m0[j+index] = 1
			}

			for j := k; j < vectorSize; j++ {
				m1[j+index] = 1
			}
		}

		diagMatrix[slots-vectorSize+k] = m0
		diagMatrix[k] = m1

		// Encoding
		matrix.LogSlots = logSlots
		matrix.Level = level
		matrix.Scale = scale
		matrix.Naive = true

		// Encode m0
		matrix.Vec[slots-vectorSize+k] = encoder.EncodeDiagonal(logSlots, level, scale, utils2.RotateComplex128Slice(m0, slots-vectorSize+k))
		// Encode m1
		matrix.Vec[k] = encoder.EncodeDiagonal(logSlots, level, scale, utils2.RotateComplex128Slice(m1, k))

	} else {
		// If N = vectorSize, the we a single rotation without masking is sufficient
		matrix.Vec[k] = rlwe2.PolyQP{}
	}

	return matrix
}

type TransposeLT struct {
	dimension  int
	levelstart int
	ckks2.PtDiagMatrix
}

func (lt *TransposeLT) StringMap() string {
	return fmt.Sprintf("0x%04o0x%04o", lt.dimension, lt.levelstart)
}

// GenTransposeDiagMatrix generates the linear transform plaintext vectors for the transpose of a square matrix.
func GenTransposeDiagMatrix(level int, scale, maxM1N2Ratio float64, dimension int, params ckks2.Parameters, encoder ckks2.Encoder) (lt TransposeLT) {

	slots := 1 << params.LogSlots()

	diagMatrix := make(map[int][]complex128)

	d2 := dimension * dimension

	for i := -dimension + 1; i < dimension; i++ {

		m := make([]complex128, slots)

		if i >= 0 {
			for j := 0; j < d2-i*dimension; j = j + dimension + 1 {
				m[i+j] = 1
			}
		} else {
			for j := -i * dimension; j < d2; j = j + dimension + 1 {
				m[j] = 1
			}
		}

		populateVector(m, d2, params.LogSlots())

		diagMatrix[(i*(dimension-1)+slots)%slots] = m
	}

	return TransposeLT{dimension, level, encoder.EncodeDiagMatrixBSGSAtLvl(level, diagMatrix, scale*params.QiFloat64(params.MaxLevel()), maxM1N2Ratio, params.LogSlots())}
}

func populateVector(m []complex128, d2, logSlots int) {

	slots := 1 << logSlots

	for k := d2; k < int(slots); k = k + d2 {

		if k+d2 > int(slots) {
			break
		}

		for j := 0; j < d2; j++ {
			m[k+j] = m[j]
		}
	}
}
