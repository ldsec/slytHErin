package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"testing"
	"time"
)

/********************************************
REGULAR MATRICES OPS
|
|
v
*********************************************/
func TestEncMult(t *testing.T) {
	//make sure that input dim*4 < 2^logSlots
	LDim := []int{92, 64}
	W0Dim := []int{64, 64}
	W1Dim := []int{64, 64}

	r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < LDim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < LDim[1]; j++ {
			fmt.Printf("%7.4f, ", L[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	W0 := make([][]float64, W0Dim[0])
	for i := range W0 {
		W0[i] = make([]float64, W0Dim[1])

		for j := range W0[i] {
			W0[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < W0Dim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < W0Dim[1]; j++ {
			fmt.Printf("%7.4f, ", W0[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	W1 := make([][]float64, W1Dim[0])
	for i := range W1 {
		W1[i] = make([]float64, W1Dim[1])

		for j := range W1[i] {
			W1[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < W1Dim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < W1Dim[1]; j++ {
			fmt.Printf("%7.4f, ", W1[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	Lmat := mat.NewDense(LDim[0], LDim[1], plainUtils.Vectorize(L, true))
	W0mat := mat.NewDense(W0Dim[0], W0Dim[1], plainUtils.Vectorize(W0, true))
	W1mat := mat.NewDense(W1Dim[0], W1Dim[1], plainUtils.Vectorize(W1, true))

	// Schemes parameters are created from scratch
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40},
		LogP:         []int{61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	if err != nil {
		panic(err)
	}

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := []int{}
	for i := 1; i < len(W0); i++ {
		rotations = append(rotations, 2*i*LDim[0])
	}

	for i := 1; i < len(W1); i++ {
		rotations = append(rotations, 2*i*LDim[0])
	}

	rotations = append(rotations, len(L))
	rotations = append(rotations, len(W0))
	rotations = append(rotations, len(W1))
	rotations = append(rotations, -len(W0)*len(L))
	rotations = append(rotations, -2*len(W0)*len(L))
	rotations = append(rotations, -len(W1)*len(L))
	rotations = append(rotations, -2*len(W1)*len(L))

	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	Box := CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}
	ctW0 := EncryptWeights(params.MaxLevel(), W0, len(L), Box)
	ctW1 := EncryptWeights(params.MaxLevel(), W1, len(L), Box)
	ctA := EncryptInput(params.MaxLevel(), L, Box)

	now := time.Now()
	B := Cipher2CMul(ctA, len(L), len(W0), ctW0, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	C := Cipher2CMul(B, len(L), len(W1), ctW1, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))
	resPt := dec.DecryptNew(C)
	resArray := ecd.DecodeSlots(resPt, params.LogSlots())
	resReal := plainUtils.ComplexToReal(resArray)[:LDim[0]*W1Dim[1]]

	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)
	var res mat.Dense
	res.Mul(&tmp, W1mat)
	fmt.Println("________________-")
	fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(plainUtils.TransposeDense(&res)), resReal))
}

func TestEncPlainMult(t *testing.T) {
	//make sure that input dim*4 < 2^logSlots
	LDim := []int{64, 64}
	W0Dim := []int{64, 64}
	W1Dim := []int{64, 64}

	//r := rand.New(rand.NewSource(0))

	r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < LDim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < LDim[1]; j++ {
			fmt.Printf("%7.4f, ", L[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	W0 := make([][]float64, W0Dim[0])
	for i := range W0 {
		W0[i] = make([]float64, W0Dim[1])

		for j := range W0[i] {
			W0[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < W0Dim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < W0Dim[1]; j++ {
			fmt.Printf("%7.4f, ", W0[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	W1 := make([][]float64, W1Dim[0])
	for i := range W1 {
		W1[i] = make([]float64, W1Dim[1])

		for j := range W1[i] {
			W1[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < W1Dim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < W1Dim[1]; j++ {
			fmt.Printf("%7.4f, ", W1[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	Lmat := mat.NewDense(LDim[0], LDim[1], plainUtils.Vectorize(L, true))
	W0mat := mat.NewDense(W0Dim[0], W0Dim[1], plainUtils.Vectorize(W0, true))
	W1mat := mat.NewDense(W1Dim[0], W1Dim[1], plainUtils.Vectorize(W1, true))

	// Schemes parameters are created from scratch
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40},
		LogP:         []int{61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})
	if err != nil {
		panic(err)
	}

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	rotations := []int{}
	for i := 1; i < len(W0); i++ {
		rotations = append(rotations, 2*i*LDim[0])
	}

	for i := 1; i < len(W1); i++ {
		rotations = append(rotations, 2*i*LDim[0])
	}

	rotations = append(rotations, len(L))
	rotations = append(rotations, len(W0))
	rotations = append(rotations, len(W1))
	rotations = append(rotations, -len(W0)*len(L))
	rotations = append(rotations, -2*len(W0)*len(L))
	rotations = append(rotations, -len(W1)*len(L))
	rotations = append(rotations, -2*len(W1)*len(L))

	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	Box := CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}
	ptW0 := EncodeWeights(params.MaxLevel(), W0, len(L), Box)
	ptW1 := EncodeWeights(params.MaxLevel(), W1, len(L), Box)
	ctA := EncryptInput(params.MaxLevel(), L, Box)

	now := time.Now()
	B := Cipher2PMul(ctA, len(L), len(W0), ptW0, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	C := Cipher2PMul(B, len(L), len(W1), ptW1, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))
	resPt := dec.DecryptNew(C)
	resArray := ecd.DecodeSlots(resPt, params.LogSlots())
	resReal := plainUtils.ComplexToReal(resArray)[:LDim[0]*W1Dim[1]]
	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)
	var res mat.Dense
	res.Mul(&tmp, W1mat)
	fmt.Println("Exp:", plainUtils.RowFlatten(plainUtils.TransposeDense(&res)))
	fmt.Println("test:", resReal)
	fmt.Println("________________-")
	fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(plainUtils.TransposeDense(&res)), resReal))
}
