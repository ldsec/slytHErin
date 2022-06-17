package multidim

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"gonum.org/v1/gonum/mat"
	"math"
)

func MatrixPrint(X mat.Matrix) {
	var empty *mat.Dense
	if X == empty {
		fmt.Printf("[]\n")
		return
	}
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
func ExplicitTranspose(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	ret := mat.NewDense(c, r, nil)
	for i := 0; i < r; i++ {
		ret.SetCol(i, m.(*mat.Dense).RawRowView(i))
	}
	return ret
}

func DebugMD(clear *mat.Dense, cipher *CiphertextBatchMatrix, innerDim, parallelBatches int, decodeParallel, transposeCipher bool, Box Ckks2Box) error {
	encoder := Box.Encoder
	decryptor := Box.Decryptor
	params := Box.Params
	resPlain2 := plainUtils.RowFlatten(clear)
	var resCipher []float64
	var resCipher2 []float64
	var err error
	if decodeParallel {
		resCipher = UnpackCipherParallel(cipher, innerDim, plainUtils.NumCols(clear), plainUtils.NumRows(clear), encoder, decryptor, params, parallelBatches)
	} else {
		resCipher = UnpackCipherSingle(cipher, innerDim, plainUtils.NumCols(clear), plainUtils.NumRows(clear), encoder, decryptor, params)
	}
	if transposeCipher {
		resCipher2 = plainUtils.RowFlatten(plainUtils.TransposeDense(mat.NewDense(plainUtils.NumCols(clear), plainUtils.NumRows(clear), resCipher)))
	} else {
		resCipher2 = plainUtils.RowFlatten(mat.NewDense(plainUtils.NumRows(clear), plainUtils.NumCols(clear), resCipher))
	}

	for i := range resPlain2 {
		fmt.Println("Test ", i, " :", resCipher2[i])
		fmt.Println("Want ", i, " :", resPlain2[i])
		fmt.Println()
		if math.Abs(resCipher2[i]-resPlain2[i]) > 0.99 {
			err = errors.New(fmt.Sprintf("Expected %f, got %f, at %d", resPlain2[i], resCipher[i], i))
			break
		}
	}
	return err
}
