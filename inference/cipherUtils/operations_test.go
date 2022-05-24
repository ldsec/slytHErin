package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math"
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
	//ct x ct
	LDim := []int{64, 26}
	W0Dim := []int{26, 23}
	W1Dim := []int{23, 10}

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
	ckksParams := ckks.DefaultParams[2]
	params, err := ckks.NewParametersFromLiteral(ckksParams)
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
	ctW1 := EncryptWeights(params.MaxLevel()-1, W1, len(L), Box)
	ctA := EncryptInput(params.MaxLevel(), L, Box)

	now := time.Now()
	B := Cipher2CMul(ctA, len(L), len(W0), len(W0[0]), ctW0, true, true, Box)

	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	C := Cipher2CMul(B, len(L), len(W1), len(W1[0]), ctW1, true, true, Box)

	fmt.Println("Done:", time.Since(now))
	resPt := dec.DecryptNew(C)
	resArray := ecd.DecodeSlots(resPt, params.LogSlots())
	resReal := plainUtils.ComplexToReal(resArray)[:LDim[0]*W1Dim[1]]

	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)
	PrintDebug(B, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(&tmp))), Box)
	CompareMatrices(B, len(L), len(W0[1]), &tmp, Box)
	var res mat.Dense
	res.Mul(&tmp, W1mat)
	fmt.Println("Final distance")
	fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(plainUtils.TransposeDense(&res)), resReal))
	PrintDebug(C, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(&res))), Box)
}

func TestEncPlainMult(t *testing.T) {
	//make sure that input dim*4 < 2^logSlots
	//ct x pt
	LDim := []int{64, 29}
	W0Dim := []int{29, 13}
	W1Dim := []int{13, 10}
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

	//ckksParams := ckks.PN14QP438
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
	ptW1 := EncodeWeights(params.MaxLevel()-1, W1, len(L), Box)
	ptW2 := EncodeWeights(params.MaxLevel()-2, W2, len(L), Box)
	ctA := EncryptInput(params.MaxLevel(), L, Box)

	now := time.Now()
	B := Cipher2PMul(ctA, len(L), len(W0), len(W0[0]), ptW0, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	C := Cipher2PMul(B, len(L), len(W1), len(W1[0]), ptW1, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))
	D := Cipher2PMul(C, len(L), len(W2), len(W2[0]), ptW2, true, true, Box)
	resPt := dec.DecryptNew(D)
	resArray := ecd.DecodeSlots(resPt, params.LogSlots())
	resReal := plainUtils.ComplexToReal(resArray)[:LDim[0]*W2Dim[1]]
	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)
	PrintDebug(B, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(&tmp))), Box)
	var tmp2 mat.Dense
	tmp2.Mul(&tmp, W1mat)
	PrintDebug(C, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(&tmp2))), Box)
	var res mat.Dense
	res.Mul(&tmp2, W2mat)
	PrintDebug(D, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.TransposeDense(&res))), Box)
	//fmt.Println("Exp:", plainUtils.RowFlatten(plainUtils.TransposeDense(&res)))
	//fmt.Println("test:", resReal)
	//fmt.Println("________________-")
	fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(plainUtils.TransposeDense(&res)), resReal))
}

func TestEvalPoly(t *testing.T) {
	//Evaluates a polynomial on ciphertext
	LDim := []int{64, 64}
	L := plainUtils.RandMatrix(LDim[0], LDim[1])
	L.Set(0, 0, 32.0)
	ckksParams := ckks.DefaultParams[2]
	params, err := ckks.NewParametersFromLiteral(ckksParams)
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
	f := func(x float64) float64 {
		return math.Log(1 + math.Exp(x))
	}
	a, b := -35.0, 35.0
	deg := 31
	approxF := ckks.Approximate(f, a, b, deg)
	fmt.Println(approxF.Coeffs)
	/*
		MatLab := []float64{-1.0040897579718860e-53, 6.2085331754358028e-40, 9.4522902777573076e-50, -5.7963804324148821e-36, -4.0131279328625271e-46, 2.4410642683332394e-32, 1.0153477706512291e-42, -6.1290204181405624e-29, -1.7039434123075587e-39, 1.0216863193793685e-25, 1.9976235851829888e-36, -1.1917424918638167e-22, -1.6781853595392470e-33, 9.9891167268766684e-20, 1.0196230261578948e-30, -6.0833342283869143e-17, -4.4658877204790776e-28, 2.6909707871865122e-14, 1.3889468322950614e-25, -8.5600457797298628e-12, -2.9800845828620543e-23, 1.9200743786780711e-09, 4.2045289670858245e-21, -2.9487406547016763e-07, -3.6043867162675355e-19, 2.9886906932909647e-05, 1.6307741516672765e-17, -1.9601130409477464e-03, -2.8618809778714450e-16, 1.0678923596705732e-01, 5.0000000000000022e-01, 7.1225856852636027e-01}
		coeffs := make([]float64, len(MatLab))
		j := len(MatLab) - 1
		for i := 0; i < len(coeffs); i++ {
			coeffs[i] = MatLab[j-i]
			//fmt.Printf("%.4e * x^%d ", relu.Coeffs[i], i)
		}
		interval := 1.0 //--> incorporate this in weight matrix to spare a level
		poly := ckks.NewPoly(plainUtils.RealToComplex(coeffs))
	*/
	slotsIndex := make(map[int][]int)
	idx := make([]int, len(plainUtils.RowFlatten(L)))
	for i := 0; i < len(plainUtils.RowFlatten(L)); i++ {
		idx[i] = i
	}
	slotsIndex[0] = idx
	//ctL := EncryptInput(params.MaxLevel(), plainUtils.MatToArray(plainUtils.MulByConst(L, 1.0/interval)), Box)
	ctL := EncryptInput(params.MaxLevel(), plainUtils.MatToArray(L), Box)
	fmt.Println("Before", ctL.Level())
	// Change of variable
	eval.MultByConst(ctL, 2/(b-a), ctL)
	eval.AddConst(ctL, (-a-b)/(b-a), ctL)
	if err := eval.Rescale(ctL, params.DefaultScale(), ctL); err != nil {
		panic(err)
	}
	ct, err := eval.EvaluatePolyVector(ctL, []*ckks.Polynomial{approxF}, ecd, slotsIndex, ctL.Scale)
	fmt.Println("After", ct.Level())
	fmt.Println("Deg", approxF.Degree())
	fmt.Println("Done... Consumed levels:", params.MaxLevel()-ct.Level())

	/*
		func(X *mat.Dense, interval float64, degree int, coeffs []float64) {
			rows, cols := X.Dims()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					v := X.At(r, c) / float64(interval)
					res := 0.0
					for deg := 0; deg < degree; deg++ {
						res += (math.Pow(v, float64(deg)) * coeffs[deg])
					}
					X.Set(r, c, res)
				}
			}
		}(L, interval, 32, coeffs)
	*/
	for i := 0; i < LDim[0]; i++ {
		for j := 0; j < LDim[1]; j++ {
			L.Set(i, j, f(L.At(i, j)))
		}
	}

	CompareMatrices(ct, LDim[0], LDim[1], L, Box)
	PrintDebug(ct, plainUtils.RealToComplex(plainUtils.Vectorize(plainUtils.MatToArray(L), true)), Box)
}

func TestBootstrap(t *testing.T) {
	//Test Bootstrap operation following lattigo examples
	LDim := []int{64, 64}
	L := plainUtils.RandMatrix(LDim[0], LDim[1])
	//crucial that parameters are conjuncted
	ckksParams := bootstrapping.DefaultParametersSparse[4].SchemeParams
	btpParams := bootstrapping.DefaultParametersSparse[4].BootstrappingParams

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

/*

 */
func TestEncodingForSmallCoeffs(t *testing.T) {
	exp := 21.0 //31 broken, 30 fine
	base := 2.0
	v := 1.0 * math.Pow(base, exp)
	//vinv := 1.0 * math.Pow(base, -exp)
	L := make([][]float64, 2)
	for i := 0; i < len(L); i++ {
		L[i] = make([]float64, 2)
		for j := 0; j < len(L[i]); j++ {
			L[i][j] = v
		}
	}
	ckksParams := ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{40, 30},
		LogP:         []int{33},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 30),
	}
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	utils.ThrowErr(err)
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})
	Box := CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}
	ctL := EncryptInput(params.MaxLevel(), L, Box)
	//eval.MultByConst(ctL, vinv, ctL)
	//eval.Rescale(ctL, params.DefaultScale(), ctL)
	eval.DropLevel(ctL, 1)
	PrintDebug(ctL, plainUtils.RealToComplex(plainUtils.RowFlatten(plainUtils.MulByConst(plainUtils.NewDense(L), 1.0))), Box)
	Lflat := plainUtils.RowFlatten(plainUtils.NewDense(L))
	ecdL := ecd.EncodeNew(Lflat, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	decdL := plainUtils.ComplexToReal(ecd.Decode(ecdL, params.LogSlots()))[:len(Lflat)]
	for i := 0; i < len(Lflat); i++ {
		fmt.Printf("test:%.32f\n", decdL[i])
		fmt.Printf("want:%.32f\n", Lflat[i])
	}
}
