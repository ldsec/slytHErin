package modelsPlain

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/data"
	"testing"
)

/*
func TestEncMult(t *testing.T) {

	LDim := []int{3, 3}
	W0Dim := []int{3, 3}
	W1Dim := []int{3, 3}

	//r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = float64(i*LDim[0] + j)
		}
	}
	fmt.Println("L:", L)
	W0 := make([][]float64, W0Dim[0])
	for i := range W0 {
		W0[i] = make([]float64, W0Dim[1])

		for j := range W0[i] {
			W0[i][j] = float64(i*W0Dim[0] + j)
		}
	}
	fmt.Println("W0:", W0)

	W1 := make([][]float64, W1Dim[0])
	for i := range W1 {
		W1[i] = make([]float64, W1Dim[1])

		for j := range W1[i] {
			W1[i][j] = float64(i*W1Dim[0] + j)
		}
	}
	fmt.Println("W1", W1)

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
		rotations = append(rotations, 2*i*len(W0))
	}

	for i := 1; i < len(W1); i++ {
		rotations = append(rotations, 2*i*len(W1))
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
	//dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})

	ctW0 := EncryptWeights(params.MaxLevel(), W0, len(L), params, ecd, enc)
	ctW1 := EncryptWeights(params.MaxLevel(), W1, len(L), params, ecd, enc)
	ctA := EncryptInput(params.MaxLevel(), L, params, ecd, enc)

	now := time.Now()
	B := Dense(ctA, len(L), len(W0), len(W0[0]), ctW0, true, true, params, eval, ecd)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	_ = Dense(B, len(L), len(W1), len(W1[0]), ctW1, true, true, params, eval, ecd)
	// -> Activate
	fmt.Println("Done:", time.Since(now))
	//resPt := dec.DecryptNew(C)
	//resArray := ecd.DecodeSlots(resPt, 14)
}
*/

func TestEvalPlain(t *testing.T) {
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 8
	inputLayerDim, _ := buildKernelMatrix(sn.Conv1.Weight, -1).Dims()
	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	dataSn.Init(batchSize, inputLayerDim)
	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil {
			break
		}
		res := sn.EvalBatchPlain(Xbatch, Y, inputLayerDim, 10)
		corrects += res.Corrects
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}

/*
func TestEvalDataEnc(t *testing.T) {
	// Schemes parameters are created from scratch
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         17,
		LogQ:         []int{60, 60, 60, 40, 40},
		LogP:         []int{61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     16,
		DefaultScale: float64(1 << 40),
	})
	if err != nil {
		fmt.Println(err)
	}

}
*/
