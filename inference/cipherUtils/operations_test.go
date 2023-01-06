package cipherUtils

import (
	"fmt"
	pU "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"testing"
)

/********************************************
REGULAR MATRICES OPS
|
|
v
*********************************************/
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
	//check that DiagMul have the rescale and add conj uncommented
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

func TestBootstrap(t *testing.T) {
	//Test Bootstrap operation following lattigo examples
	LDim := []int{64, 64}
	//L := pU.RandMatrix(LDim[0], LDim[1])
	v := make([]float64, LDim[0]*LDim[1])
	for i := 0; i < LDim[0]*LDim[1]; i++ {
		v[i] = rand.Float64()
	}
	L := mat.NewDense(LDim[0], LDim[1], v)

	//ckksParams := bootstrapping.N16QP1553H192H32.SchemeParams
	//btpParams := bootstrapping.N16QP1553H192H32.BootstrappingParams

	ckksParams := bootstrapping.N16QP1546H192H32.SchemeParams
	btpParams := bootstrapping.N16QP1546H192H32.BootstrappingParams

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
	ct2 := btp.Bootstrapp(ctL)
	fmt.Println("Done")
	fmt.Println("Scale", ct2.Scale-params.DefaultScale())

	//fmt.Println("Precision of ciphertext vs. Bootstrapp(ciphertext)")
	//PrintDebug(ct2, pU.RealToComplex(pU.Vectorize(pU.MatToArray(L), false)), 0.01, Box)
}

/*
func TestBootstrapDistributed(t *testing.T) {
	PARTIES := []int{5, 10}
	PARAMS := []ckks.ParametersLiteral{ckks.PN15QP880, ckks.ParametersLiteral{
		LogN:         15,
		LogSlots:     14,
		LogQ:         []int{50, 37, 37, 37, 37, 37, 37, 37, 37}, //31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
		LogP:         []int{43, 43, 43},
		DefaultScale: 1 << 37,
		Sigma:        rlwe.DefaultSigma,
		RingType:     ring.Standard,
	}}
	L := pU.RandMatrix(64, 64)
	L.Set(0, 0, 30)
	for _, ckksParams := range PARAMS[1:] { //logN 14 works fine with everyone
		for _, parties := range PARTIES { //parties = 3 is fine always
			fmt.Printf("Test: parties %d, params %d\n\n", parties, ckksParams.LogN)
			params, _ := ckks.NewParametersFromLiteral(ckksParams)
			crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})
			skShares, skP, pkP, _ := distributed.DummyEncKeyGen(params, crs, parties)
			rlk := distributed.DummyRelinKeyGen(params, crs, skShares)
			decP := ckks.NewDecryptor(params, skP)
			Box := CkksBox{
				Params:       params,
				Encoder:      ckks.NewEncoder(params),                                            //public
				Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil}), //from parties
				Decryptor:    decP,                                                               //from parties for debug
				Encryptor:    ckks.NewEncryptor(params, pkP),                                     //from parties
				BootStrapper: nil,
			}
			//Create distributed parties
			localhost := "127.0.0.1"
			partiesAddr := make([]string, parties)
			for i := 0; i < parties; i++ {
				if i == 0 {
					partiesAddr[i] = localhost + ":" + strconv.Itoa(8000)
				} else {
					partiesAddr[i] = localhost + ":" + strconv.Itoa(8080+i)
				}
			}
			master, err := distributed.NewLocalMaster(skShares[0], pkP, params, parties, partiesAddr, Box, 1)
			utils.ThrowErr(err)
			players := make([]*distributed.LocalPlayer, parties-1)
			//start players
			for i := 0; i < parties-1; i++ {
				players[i], err = distributed.NewLocalPlayer(skShares[i+1], pkP, params, i+1, partiesAddr[i+1])
				go players[i].Listen()
				utils.ThrowErr(err)
			}
			minLevel, _, _ := dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q())
			ctL := EncryptInput(params.MaxLevel(), pU.MatToArray(L), Box)
			Box.Evaluator.DropLevel(ctL, params.MaxLevel()-minLevel)
			ctBtp, err := master.InitProto(distributed.REFRESH, nil, ctL, 0)
			utils.ThrowErr(err)
			PrintDebug(ctBtp, pU.RealToComplex(pU.RowFlatten(L)), 0.001, Box)
			master.InitProto(distributed.END, nil, nil, 0)
			time.Sleep(1000 * time.Millisecond) //wait for stop
		}
	}
}
*/
