package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
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
	//make sure that input dim*2 < 2^logSlots
	//ct x ct
	LDim := []int{8, 4}
	W0Dim := []int{4, 2}
	W1Dim := []int{2, 4}

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
	/*
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
	*/
	rotations = GenRotations(len(L), 2, []int{len(W0), len(W1)}, []int{len(W0[0]), len(W1[0])}, params, nil)
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
	//ct x pt
	LDim := []int{64, 29}
	W0Dim := []int{29, 65}
	W1Dim := []int{65, 10}
	W2Dim := []int{10, 10}

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

	W2 := make([][]float64, W2Dim[0])
	for i := range W2 {
		W2[i] = make([]float64, W2Dim[1])

		for j := range W2[i] {
			W2[i][j] = r.NormFloat64()
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
	W2mat := mat.NewDense(W2Dim[0], W2Dim[1], plainUtils.Vectorize(W2, true))

	// Schemes parameters are created from scratc
	//ckksParams := bootstrapping.DefaultCKKSParameters[0]
	//ckksParams := ckks.PN15QP880
	//params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral(ckksParams))
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 60, 60, 40, 40, 40, 40, 40, 40},
		LogP:         []int{61, 61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 40),
	})

	utils.ThrowErr(err)
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	/*
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
	*/
	rotations := GenRotations(len(L), 3, []int{len(W0), len(W1), len(W2)}, []int{len(W0[0]), len(W1[0]), len(W2[0])}, params, nil)

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
	ptW2 := EncodeWeights(params.MaxLevel(), W2, len(L), Box)
	ctA := EncryptInput(params.MaxLevel(), L, Box)

	now := time.Now()
	B := Cipher2PMul(ctA, len(L), len(W0), ptW0, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	C := Cipher2PMul(B, len(L), len(W1), ptW1, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))
	D := Cipher2PMul(C, len(L), len(W2), ptW2, true, true, Box)
	resPt := dec.DecryptNew(D)
	resArray := ecd.DecodeSlots(resPt, params.LogSlots())
	resReal := plainUtils.ComplexToReal(resArray)[:LDim[0]*W2Dim[1]]
	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)
	var tmp2 mat.Dense
	tmp2.Mul(&tmp, W1mat)
	var res mat.Dense
	res.Mul(&tmp2, W2mat)
	//fmt.Println("Exp:", plainUtils.RowFlatten(plainUtils.TransposeDense(&res)))
	//fmt.Println("test:", resReal)
	//fmt.Println("________________-")
	fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(plainUtils.TransposeDense(&res)), resReal))
}

/*
func TestEvalPoly(t *testing.T) {
	//Evaluates a polynomial on ciphertext
	LDim := []int{64, 64}
	L := plainUtils.RandMatrix(LDim[0], LDim[1])

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

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil})
	Box := CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}
	//relu approx
	coeffs := []float64{1.1155, 5.0, 4.4003} //degree 2
	interval := 10.0                         //--> incorporate this in weight matrix to spare a level
	poly := ckks.NewPoly(plainUtils.RealToComplex(coeffs))

	ctL := EncryptInput(params.MaxLevel(), plainUtils.MatToArray(L), Box)
	eval.MultByConst(ctL, float64(1/interval), ctL)
	if err := eval.Rescale(ctL, params.DefaultScale(), ctL); err != nil {
		panic(err)
	}
	ct, err := eval.EvaluatePoly(ctL, poly, ctL.Scale)
	fmt.Println("Done... Consumed levels:", params.MaxLevel()-ct.Level())

	sn := new(modelsPlain.SimpleNet)
	sn.InitActivation()
	sn.ActivatePlain(L)

	CompareMatrices(ct, LDim[0], LDim[1], L, Box)
	PrintDebug(ct, plainUtils.RealToComplex(plainUtils.Vectorize(plainUtils.MatToArray(L), true)), Box)
}

func TestBootstrap(t *testing.T) {
	//Test Bootstrap operation following lattigo examples
	LDim := []int{64, 64}
	L := plainUtils.RandMatrix(LDim[0], LDim[1])
	//crucial that parameters are conjuncted
	ckksParams := bootstrapping.DefaultCKKSParameters[0]
	btpParams := bootstrapping.DefaultParameters[0]

	params, err := ckks.NewParametersFromLiteral(ckksParams)
	if err != nil {
		panic(err)
	}
	rotations := btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
	if err != nil {
		panic(err)
	}

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	if err != nil {
		panic(err)
	}
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil})
	Box := CkksBox{
		Params:       params,
		Encoder:      ecd,
		Evaluator:    eval,
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: btp,
	}
	//relu approx

	ctL := EncryptInput(params.MaxLevel(), plainUtils.MatToArray(L), Box)

	// Bootstrap the ciphertext (homomorphic re-encryption)
	// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
	// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
	// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
	// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
	fmt.Println()
	fmt.Println("Bootstrapping...")
	ct2 := btp.Bootstrapp(ctL)
	fmt.Println("Done")

	fmt.Println("Precision of ciphertext vs. Bootstrapp(ciphertext)")
	CompareMatrices(ct2, LDim[0], LDim[1], L, Box)
	PrintDebug(ct2, plainUtils.RealToComplex(plainUtils.Vectorize(plainUtils.MatToArray(L), true)), Box)
}
*/
