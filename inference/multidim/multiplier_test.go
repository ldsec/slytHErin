package multidim

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
	"github.com/stretchr/testify/require"
)

var minPrec float64 = 15.0

var printPrecisionStats = flag.Bool("print-precision", false, "print precision stats")

func TestSingleMatrixOps(t *testing.T) {

	rand.Seed(time.Now().UnixNano())

	// Schemes parameters are created from scratch
	params, err := ckks2.NewParametersFromLiteral(ckks2.ParametersLiteral{
		LogN:     13,
		LogQ:     []int{50, 35, 35, 35},
		LogP:     []int{61, 61},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 12,
		Scale:    float64(1 << 35),
	})
	if err != nil {
		panic(err)
	}

	encoder := ckks2.NewEncoder(params)
	kgen := ckks2.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)
	encryptor := ckks2.NewEncryptor(params, kgen.GenPublicKey(sk))
	decryptor := ckks2.NewDecryptor(params, sk)
	eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: nil})

	// Size of the matrices (dxd)
	rows := 8
	cols := 8

	t.Run("RotateCols", func(t *testing.T) {

		m, _, ct := GenTestVectors(rows, cols, params, encoder, encryptor)

		alpha := params.Alpha()
		beta := int(math.Ceil(float64(ct.Level()+1) / float64(alpha)))

		polyDecompQPA := make([]rlwe2.PolyQP, beta)

		for i := 0; i < beta; i++ {
			polyDecompQPA[i].Q = params.RingQ().NewPolyLvl(ct.Level())
			polyDecompQPA[i].P = params.RingP().NewPoly()
		}

		eval.GetKeySwitcher().DecomposeNTT(ct.Level(), alpha-1, alpha, ct.Value[1], polyDecompQPA)

		res := ckks2.NewCiphertext(params, 1, ct.Level(), ct.Scale)

		for k := 1; k < rows; k++ {

			t.Run(fmt.Sprintf("k=%d/", k), func(t *testing.T) {

				level := params.MaxLevel()

				diagMatrix := GenSubVectorRotationMatrix(level, params.QiFloat64(level), rows, k, params.LogSlots(), encoder)
				rotations := params.RotationsForDiagMatrixMult(diagMatrix)
				rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)

				eval0 := eval.WithKey(rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})

				for j := range m {
					m[j].RotateCols(1)
				}

				eval0.MultiplyByDiagMatrix(ct, diagMatrix, polyDecompQPA, res)

				VerifyTestVectors(params, encoder, decryptor, m, res, t)
			})
		}
	})

	t.Run("RotateRows", func(t *testing.T) {
		m, _, ct := GenTestVectors(rows, cols, params, encoder, encryptor)

		alpha := params.Alpha()
		beta := int(math.Ceil(float64(ct.Level()+1) / float64(alpha)))

		polyDecompQPA := make([]rlwe2.PolyQP, beta)

		for i := 0; i < beta; i++ {
			polyDecompQPA[i].Q = params.RingQ().NewPolyLvl(ct.Level())
			polyDecompQPA[i].P = params.RingP().NewPoly()
		}

		eval.GetKeySwitcher().DecomposeNTT(ct.Level(), alpha-1, alpha, ct.Value[1], polyDecompQPA)

		res := ckks2.NewCiphertext(params, 1, ct.Level(), ct.Scale)

		for k := 1; k < rows; k++ {

			t.Run(fmt.Sprintf("k=%d/", k), func(t *testing.T) {

				level := params.MaxLevel()

				diagMatrix := GenSubVectorRotationMatrix(level, params.QiFloat64(level), rows*rows, k*rows, params.LogSlots(), encoder)
				rotations := params.RotationsForDiagMatrixMult(diagMatrix)
				rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)

				eval0 := eval.WithKey(rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})

				for j := range m {
					m[j].RotateRows(1)
				}

				eval0.MultiplyByDiagMatrix(ct, diagMatrix, polyDecompQPA, res)

				VerifyTestVectors(params, encoder, decryptor, m, res, t)
			})
		}
	})

	t.Run("PermuteRows/", func(t *testing.T) {
		m, _, ct := GenTestVectors(rows, cols, params, encoder, encryptor)

		level := params.MaxLevel()

		diagMatrix := GenPermuteRowsMatrix(level, params.QiFloat64(level), 16.0, rows, params.LogSlots(), encoder)
		rotations := params.RotationsForDiagMatrixMult(diagMatrix)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)

		eval0 := eval.WithKey(rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})

		for j := range m {
			m[j].PermuteRows()
		}

		ct = eval0.LinearTransformNew(ct, diagMatrix)[0]

		//PrintDebug(ct, rows, cols, params, encoder, decryptor)

		VerifyTestVectors(params, encoder, decryptor, m, ct, t)

	})

	t.Run("PermuteCols/", func(t *testing.T) {
		m, _, ct := GenTestVectors(rows, cols, params, encoder, encryptor)

		level := params.MaxLevel()

		diagMatrix := GenPermuteColsMatrix(level, params.QiFloat64(level), 16.0, rows, params.LogSlots(), encoder)
		rotations := params.RotationsForDiagMatrixMult(diagMatrix)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)

		eval0 := eval.WithKey(rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})

		for j := range m {
			m[j].PermuteCols()
		}

		//PrintDebug(ct, d, params, encoder, decryptor)

		ct = eval0.LinearTransformNew(ct, diagMatrix)[0]

		//PrintDebug(ct, d, params, encoder, decryptor)

		VerifyTestVectors(params, encoder, decryptor, m, ct, t)

	})
}

func EncodeMatrices(values []complex128, pt *ckks2.Plaintext, m []*Matrix, encoder ckks2.Encoder, logSlots int) {
	d := m[0].Rows() * m[0].Cols()
	for i := 0; i < utils2.MinInt(len(m), (1<<logSlots)/d); i++ {
		for j, c := range m[i].M {
			values[i*d+j] = c
		}
	}
	encoder.Encode(pt, values, logSlots)
}

func MatricesToVectorNew(m []*Matrix, params ckks2.Parameters) (values []complex128) {
	values = make([]complex128, params.Slots())
	d := m[0].Rows() * m[0].Cols()
	for i := 0; i < utils2.MinInt(len(m), int(params.Slots())/d); i++ {
		copy(values[i*d:(i+1)*d], m[i].M)
	}
	return
}

func GenTestVectors(rows, cols int, params ckks2.Parameters, encoder ckks2.Encoder, encryptor ckks2.Encryptor) (m []*Matrix, pt *ckks2.Plaintext, ct *ckks2.Ciphertext) {

	m = GenRandomComplexMatrices(rows, cols, int(params.Slots())/(rows*cols))

	values := MatricesToVectorNew(m, params)

	pt = encoder.EncodeNew(values, params.LogSlots())

	ct = encryptor.EncryptNew(pt)

	return m, pt, ct
}

func VerifyTestVectors(params ckks2.Parameters, encoder ckks2.Encoder, decryptor ckks2.Decryptor, m []*Matrix, element interface{}, t *testing.T) {

	precStats := ckks2.GetPrecisionStats(params, encoder, decryptor, MatricesToVectorNew(m, params), element, params.LogSlots(), 0)

	if *printPrecisionStats {
		t.Log(precStats.String())
	}

	require.GreaterOrEqual(t, precStats.MeanPrecision.Real, minPrec)
	require.GreaterOrEqual(t, precStats.MeanPrecision.Imag, minPrec)

}

func PrintDebug(ct *ckks2.Ciphertext, rows, cols int, params ckks2.Parameters, encoder ckks2.Encoder, decryptor ckks2.Decryptor) {

	valuesHave := encoder.Decode(decryptor.DecryptNew(ct), params.LogSlots())

	maxPrint := 1 //params.Slots() / (rows * cols)

	if maxPrint > 4 {
		maxPrint = 4
	}
	for k := 0; k < maxPrint; k++ {

		index := k * rows * cols

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				fmt.Printf("%7.4f ", real(valuesHave[index+i*rows+j]))
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}
}
