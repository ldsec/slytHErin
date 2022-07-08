package cipherUtils

import (
	"fmt"
	pU "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"testing"
)

/********************************************
REGULAR MATRICES OPS
|
|
v
*********************************************/

func TestMultiplication(t *testing.T) {
	X := pU.RandMatrix(41, 98)
	W := pU.RandMatrix(98, 80)
	//X := pU.MatrixForDebug(3, 3)
	//W := pU.MatrixForDebug(3, 3)
	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	scale := params.DefaultScale()

	Box := NewBox(params)
	Box = BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, pU.NumRows(X), pU.NumCols(X), 1, []int{pU.NumRows(W)}, []int{pU.NumCols(W)})

	t.Run("Test/C2P", func(t *testing.T) {
		Xenc := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
		Wenc := EncodeWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), Box)
		ops := make([]ckks.Operand, len(Wenc))
		for i := range Wenc {
			ops[i] = Wenc[i]
		}
		Renc := DiagMul(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), ops, true, true, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
	})

	t.Run("Test/C2C", func(t *testing.T) {
		Xenc := EncryptInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
		Wenc := EncryptWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), Box)
		ops := make([]ckks.Operand, len(Wenc))
		for i := range Wenc {
			ops[i] = Wenc[i]
		}
		Renc := DiagMul(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), ops, true, true, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		//we need to tranpose the plaintext result according to the diagonalized multiplication algo
		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
	})

	t.Run("Test/P2C", func(t *testing.T) {
		Xenc := EncodeInput(params.MaxLevel(), scale, pU.MatToArray(X), Box)
		Wenc := EncryptWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), Box)
		ops := make([]ckks.Operand, len(Wenc))
		for i := range Wenc {
			ops[i] = Wenc[i]
		}
		Xenc = PrepackClearText(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), Box)
		Renc := DiagMulPt(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), ops, false, true, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		//we need to tranpose the plaintext result according to the diagonalized multiplication algo
		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)

	})

}

func TestBootstrap(t *testing.T) {
	//Test Bootstrap operation following lattigo examples
	LDim := []int{64, 64}
	L := pU.RandMatrix(LDim[0], LDim[1])
	//crucial that parameters are conjuncted
	ckksParams := bootstrapping.DefaultParametersSparse[4].SchemeParams
	btpParams := bootstrapping.DefaultParametersSparse[4].BootstrappingParams

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
	ct2 := btp.Bootstrapp(ctL)
	fmt.Println("Done")

	fmt.Println("Precision of ciphertext vs. Bootstrapp(ciphertext)")

	PrintDebug(ct2, pU.RealToComplex(pU.Vectorize(pU.MatToArray(L), true)), 0.001, Box)
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
