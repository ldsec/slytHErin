package multidim

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
	"time"
)

func Test_Multiplication_Single(t *testing.T) {
	ckksParams := ckks2.ParametersLiteral{
		LogN:     13,
		LogQ:     []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 218 for LogN 13
		LogP:     []int{33},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 12,
		Scale:    float64(1 << 26),
	}
	params, _ := ckks2.NewParametersFromLiteral(ckksParams)

	A := plainUtils.RandMatrix(64, 64)
	B := plainUtils.RandMatrix(64, 30)
	C := plainUtils.RandMatrix(30, 10)

	dim := 8

	t.Run("Test/Mult/Plain/Single", func(t *testing.T) {
		Apacked := PackMatrixSingle(A, dim)
		Bpacked := PackMatrixSingle(B, dim)
		Cpacked := PackMatrixSingle(C, dim)

		var tmp mat.Dense
		var res mat.Dense
		tmp.Mul(A, B)
		res.Mul(&tmp, C)

		Apacked.Mul(Apacked, Bpacked)
		Cpacked.Mul(Apacked, Cpacked)

		resFromPack := UnpackMatrixSingle(Cpacked, dim, plainUtils.NumRows(&res), plainUtils.NumCols(&res))
		resFromDense := plainUtils.Vectorize(plainUtils.MatToArray(&res), true)
		fmt.Println("Pack:", len(resFromPack), "dense", len(resFromDense))
		for i := range resFromDense {
			fmt.Println("Test:", resFromPack[i])
			fmt.Println("Want:", resFromDense[i])
			require.LessOrEqual(t, math.Abs(resFromPack[i]-resFromDense[i]), 1e-10)
		}
	})

	t.Run("Test/Mult/Enc/Single", func(t *testing.T) {
		Apacked := PackMatrixSingle(A, dim)
		Bpacked := PackMatrixSingle(B, dim)
		Cpacked := PackMatrixSingle(C, dim)

		encoder := ckks2.NewEncoder(params)

		// Keys
		kgen := ckks2.NewKeyGenerator(params)
		sk, _ := kgen.GenKeyPair()

		// Relinearization key
		rlk := kgen.GenRelinearizationKey(sk, 2)

		// Decryptor
		decryptor := ckks2.NewDecryptor(params, sk)

		lvl_W0 := params.MaxLevel()
		mmLiteral := MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_1 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		transposeLT_1 := GenTransposeDiagMatrix(params.MaxLevel(), 1.0, 4.0, dim, params, encoder)
		mmLiteral = MatrixMultiplicationLiteral{
			Dimension:   dim,
			LevelStart:  lvl_W0 - 3,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		mm_2 := NewMatrixMultiplicatonFromLiteral(params, mmLiteral, encoder)
		transposeLT_2 := GenTransposeDiagMatrix(params.MaxLevel()-3, 1.0, 4.0, dim, params, encoder)

		// Rotation-keys generation
		rotations := mm_1.Rotations(params)
		rotations = append(rotations, mm_2.Rotations(params)...)
		rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_1.PtDiagMatrix)...)
		rotations = append(rotations, params.RotationsForDiagMatrixMult(transposeLT_2.PtDiagMatrix)...)
		rotKeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
		eval := ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rotKeys})
		ppm := NewPackedMatrixMultiplier(params, dim, utils2.MaxInt(Apacked.rows, Bpacked.rows), utils2.MaxInt(Apacked.cols, Bpacked.cols), eval)
		ppm.AddMatrixOperation(mm_1)
		ppm.AddMatrixOperation(transposeLT_1)
		ppm.AddMatrixOperation(mm_2)
		ppm.AddMatrixOperation(transposeLT_2)
		batchEncryptor := NewBatchEncryptor(params, sk)

		ctA := batchEncryptor.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Apacked)
		ctB := batchEncryptor.EncodeAndEncrypt(lvl_W0, params.Scale(), Bpacked)

		ctC := batchEncryptor.EncodeAndEncrypt(lvl_W0-3, params.Scale(), Cpacked)

		Apacked.Mul(Apacked, Bpacked)
		Cpacked.Mul(Apacked, Cpacked)

		start := time.Now()
		ctTmp := AllocateCiphertextBatchMatrix(ctA.Rows(), ctB.Cols(), dim, ctA.Level()-2, params)
		ppm.MulSquareMatricesPacked(ctA, ctB, dim, ctTmp)
		ctRes := AllocateCiphertextBatchMatrix(ctTmp.Rows(), ctC.Cols(), dim, ctTmp.Level()-2, params)
		fmt.Println("level: ", ctTmp.Level())
		ppm.MulSquareMatricesPacked(ctTmp, ctC, dim, ctRes)
		stop := time.Since(start)
		fmt.Println("Done ", stop)

		resPlain := UnpackMatrixSingle(Cpacked, dim, plainUtils.NumRows(A), plainUtils.NumCols(C))
		resCipher := UnpackCipherSingle(ctRes, dim, plainUtils.NumRows(A), plainUtils.NumCols(C), encoder, decryptor, params)
		for i := range resPlain {
			fmt.Println("Test:", resCipher[i])
			fmt.Println("Want:", resPlain[i])
			require.LessOrEqual(t, math.Abs(resPlain[i]-resCipher[i]), 1e-2)
		}
	})
}
