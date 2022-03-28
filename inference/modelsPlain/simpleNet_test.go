package modelsPlain

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"testing"
	"time"
)

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
	for i := 1; i < (len(W0)+1)>>1; i++ {
		rotations = append(rotations, 2*i*LDim[0])
	}

	for i := 1; i < (len(W1)+1)>>1; i++ {
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
	Box := cipherUtils.CkksBox{
		Params:    params,
		Encoder:   ecd,
		Evaluator: eval,
		Decryptor: dec,
		Encryptor: enc,
	}
	ctW0 := cipherUtils.EncryptWeights(params.MaxLevel(), W0, len(L), Box)
	ctW1 := cipherUtils.EncryptWeights(params.MaxLevel(), W1, len(L), Box)
	ctA := cipherUtils.EncryptInput(params.MaxLevel(), L, Box)

	now := time.Now()
	B := cipherUtils.Cipher2CMul(ctA, len(L), len(W0), ctW0, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	now = time.Now()
	C := cipherUtils.Cipher2CMul(B, len(L), len(W1), ctW1, true, true, Box)
	// -> Activate
	fmt.Println("Done:", time.Since(now))
	resPt := dec.DecryptNew(C)
	resArray := ecd.DecodeSlots(resPt, 14)
	resReal := plainUtils.ComplexToReal(resArray)[:(len(L) * len(W1[0]))]
	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)
	var res mat.Dense
	res.Mul(&tmp, W1mat)
	fmt.Println("________________-")
	fmt.Println(plainUtils.RowFlatten(&res))
	fmt.Println("________________-")
	fmt.Println(resReal)

}

func TestEvalPlain(t *testing.T) {
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 8
	inputLayerDim, _ := buildKernelMatrix(sn.Conv1.Weight).Dims()
	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	dataSn.Init(batchSize)
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

func TestEvalPlainBlocks(t *testing.T) {
	//leverages matrix block arithmetics and concurrent execution
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 128
	inputLayerDim, _ := buildKernelMatrix(sn.Conv1.Weight).Dims()
	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	dataSn.Init(batchSize)
	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil {
			break
		}
		res := sn.EvalBatchPlainBlocks(Xbatch, Y, inputLayerDim, 10)
		corrects += res.Corrects
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}

func TestEvalDataEnc(t *testing.T) {
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 8
	conv1M := buildKernelMatrix(sn.Conv1.Weight)
	inputLayerDim := plainUtils.NumRows(conv1M)
	bias1M := buildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)
	pool1M := buildKernelMatrix(sn.Pool1.Weight)
	bias2M := buildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)
	pool2M := buildKernelMatrix(sn.Pool2.Weight)
	bias3M := buildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	err := dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}
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
		fmt.Println(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	weights := []*mat.Dense{conv1M, pool1M, pool2M}
	bias := []*mat.Dense{bias1M, bias2M, bias3M}
	//init rotations
	rotations := []int{}
	rotations = append(rotations, batchSize)
	for w := range weights {
		weight := weights[w]
		rows := plainUtils.NumRows(weight)
		for i := 1; i < rows; i++ {
			rotations = append(rotations, 2*i*rows)
		}
	}
	rotations = append(rotations, batchSize)
	for w := range weights {
		weight := weights[w]
		rows := plainUtils.NumRows(weight)
		rotations = append(rotations, rows)
	}
	for w := range weights {
		weight := weights[w]
		rows := plainUtils.NumRows(weight)
		rotations = append(rotations, rows)
		rotations = append(rotations, -rows*batchSize)
		rotations = append(rotations, -2*rows*batchSize)
	}
	//rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	Box := cipherUtils.CkksBox{
		Params:    params,
		Encoder:   ckks.NewEncoder(params),
		Evaluator: ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil}),
		Decryptor: dec,
		Encryptor: enc,
	}
	corrects := 0
	tot := 0
	for true {
		Xbatch, _, err := dataSn.Batch()
		Xenc := cipherUtils.EncryptInput(params.MaxLevel(), Xbatch, Box)
		if err != nil {
			//dataset completed
			break
		}
		_ = sn.EvalBatchEncrypted(Xbatch, Xenc, weights, bias, Box, inputLayerDim, true)
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))

}
