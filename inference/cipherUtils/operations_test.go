package cipherUtils

import (
	"fmt"
	pU "github.com/ldsec/slytHErin/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"testing"
	"time"
)

/*
*******************************************
REGULAR MATRICES OPS
|
|
v
********************************************
*/
func TestPrepackClearText(t *testing.T) {
	X := pU.RandMatrix(40, 90)
	W := pU.RandMatrix(90, 90)
	//X := pU.MatrixForDebug(3, 3)
	//W := pU.MatrixForDebug(3, 3)

	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
		LogP:         []int{60, 60},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 30),
	})
	scale := params.DefaultScale()

	Box := NewBox(params)
	Wenc := EncryptWeights(params.MaxLevel(), pU.MatToArray(W), 40, 90, Box)
	Box = BoxWithRotations(Box, Wenc.GetRotations(params), false, nil)
	Xenc := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
	Xpt := EncodeInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)

	Prepack(Xenc, 40, 90, 90, Box.Evaluator)
	Xpt = PrepackClearText(Xpt, 40, 90, 90, Box)

	r1 := Box.Encoder.Decode(Xpt, Box.Params.LogSlots())
	PrintDebug(Xenc, r1, 0.001, Box)
}

func TestRotatePlaintext(t *testing.T) {
	X := pU.RandMatrix(40, 90)

	//X := pU.MatrixForDebug(3, 3)
	//W := pU.MatrixForDebug(3, 3)

	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
		LogP:         []int{60, 60},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 30),
	})
	scale := params.DefaultScale()

	Box := NewBox(params)
	rotations := make([]int, 20)
	for i := range rotations {
		rotations[i] = 2 * i * 40
	}
	Box = BoxWithRotations(Box, rotations, false, nil)
	Xenc := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
	Xpt := EncodeInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)

	rotPt := RotatePlaintext(Xpt, rotations, Box)
	rotCt := Box.Evaluator.RotateHoistedNew(Xenc, rotations)

	for i, r := range rotCt {
		r1 := Box.Encoder.Decode(rotPt[i], Box.Params.LogSlots())
		PrintDebug(r, r1, 0.1, Box)
	}
}

func TestMultiplication(t *testing.T) {
	//check that DiagMul have the rescale and add conj uncommented in multiplier.go
	//Why? We use a specific optimization called late rescaling for block operations
	X := pU.RandMatrix(40, 90)
	W := pU.RandMatrix(90, 90)
	//X := pU.MatrixForDebug(3, 3)
	//W := pU.MatrixForDebug(3, 3)

	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
		LogP:         []int{60, 60},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 30),
	})
	scale := params.DefaultScale()

	Box := NewBox(params)

	t.Run("Test/C2P", func(t *testing.T) {
		Xenc := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
		Wenc := EncodeWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), pU.NumCols(X), Box)
		Box = BoxWithRotations(Box, Wenc.GetRotations(params), false, nil)

		Renc := DiagMulCt(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), Wenc, true, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
	})

	t.Run("Test/C2C", func(t *testing.T) {
		Xenc := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
		Wenc := EncodeWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), pU.NumCols(X), Box)
		Box = BoxWithRotations(Box, Wenc.GetRotations(params), false, nil)

		Renc := DiagMulCt(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), Wenc, true, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
	})

	t.Run("Test/P2C", func(t *testing.T) {
		Xenc := EncodeInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
		Xenc = PrepackClearText(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), Box)
		Wenc := EncryptWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), pU.NumCols(X), Box)
		Box = BoxWithRotations(Box, Wenc.GetRotations(params), false, nil)

		Renc := DiagMulPt(Xenc, pU.NumRows(X), Wenc, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
	})

}

func TestNewObliviousDec(t *testing.T) {
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
		LogP:         []int{60, 60},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 30),
	})
	scale := params.DefaultScale()
	BoxC := NewBox(params)
	BoxS := NewBox(params)
	X := pU.Vectorize(pU.MatToArray(pU.RandMatrix(40, 90)), true)
	Xenc := BoxS.Encryptor.EncryptNew(BoxS.Encoder.EncodeNew(X, params.MaxLevel(), scale, params.LogSlots()))
	mask := BoxC.Encryptor.EncryptNew(BoxC.Encoder.EncodeNew([]float64{0.0}, params.MaxLevel(), scale, params.LogSlots()))
	mask_c0 := mask.Ciphertext.Value[0]
	mask_c1 := mask.Ciphertext.Value[1]
	params.RingQ().Add(Xenc.Value[0], mask_c0, Xenc.Value[0])
	Xdec := BoxS.Decryptor.DecryptNew(Xenc)
	params.RingQ().MulCoeffsMontgomery(mask_c1, BoxC.Sk.Value.Q, mask_c1)
	params.RingQ().Add(Xdec.Value, mask_c1, Xdec.Value)
	Xunmasked := BoxC.Encoder.Decode(Xdec, params.LogSlots())
	for i := 0; i < len(X); i++ {
		fmt.Printf("%f vs %f\n", X[i], Xunmasked[i])
	}
}

func TestBootstrap(t *testing.T) {
	//Test Bootstrap operation following lattigo examples
	//5.126327023s for N15
	LDim := []int{64, 64}
	//L := pU.RandMatrix(LDim[0], LDim[1])
	v := make([]float64, LDim[0]*LDim[1])
	for i := 0; i < LDim[0]*LDim[1]; i++ {
		v[i] = rand.Float64()
	}
	L := mat.NewDense(LDim[0], LDim[1], v)

	//ckksParams := bootstrapping.N16QP1553H192H32.SchemeParams
	//btpParams := bootstrapping.N16QP1553H192H32.BootstrappingParams

	ckksParams := bootstrapping.N15QP768H192H32.SchemeParams
	btpParams := bootstrapping.N15QP768H192H32.BootstrappingParams

	params, err := ckks.NewParametersFromLiteral(ckksParams)
	scale := params.DefaultScale()
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
	evk := bootstrapping.GenEvaluationKeys(btpParams, params, sk)
	btp, err := bootstrapping.NewBootstrapper(params, btpParams, evk)
	if err != nil {
		panic(err)
	}
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	Box := CkksBox{
		Params:       params,
		Encoder:      ecd,
		Evaluator:    eval,
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: btp,
	}
	//relu approx

	ctL := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(L), Box)

	// Bootstrap the ciphertext (homomorphic re-encryption)
	// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
	// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
	// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
	// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
	fmt.Println()
	fmt.Println("Bootstrapping...")
	fmt.Println("Scale:", ctL.Scale-params.DefaultScale())
	start := time.Now()
	ct2 := btp.Bootstrapp(ctL)
	end := time.Since(start).Seconds()
	fmt.Println("Done")
	fmt.Println("Scale", ct2.Scale-params.DefaultScale())
	fmt.Println("End: ", end)
	fmt.Println("Precision of ciphertext vs. Bootstrapp(ciphertext)")
	PrintDebug(ct2, pU.RealToComplex(pU.Vectorize(pU.MatToArray(L), false)), 0.01, Box)
}
