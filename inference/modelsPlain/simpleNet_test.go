package modelsPlain

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"testing"
)

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
	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	dataSn.Init(batchSize)
	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil {
			break
		}
		res := sn.EvalBatchPlainBlocks(Xbatch, Y, 10)
		corrects += res.Corrects
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}

func TestEvalDataEncModelClear(t *testing.T) {
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 1024
	conv1M := buildKernelMatrix(sn.Conv1.Weight)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := buildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)
	pool1M := buildKernelMatrix(sn.Pool1.Weight)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := buildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)
	pool2M := buildKernelMatrix(sn.Pool2.Weight)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := buildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	err := dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}
	ckksParams := bootstrapping.DefaultCKKSParameters[4]
	btpParams := bootstrapping.DefaultParameters[4]
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	if err != nil {
		panic(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	weights := []*mat.Dense{conv1M, pool1M, pool2M}
	rowsW := make([]int, len(weights))
	colsW := make([]int, len(weights))
	for w := range weights {
		rowsW[w], colsW[w] = weights[w].Dims()
	}
	bias := []*mat.Dense{bias1M, bias2M, bias3M}
	//init rotations
	rotations := cipherUtils.GenRotations(batchSize/32, len(weights), rowsW, colsW, params, &btpParams)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	utils.ThrowErr(err)
	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: btp,
	}
	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		Xenc, _ := cipherUtils.NewEncInput(Xbatch, 32, 29, Box)
		if err != nil {
			//dataset completed
			break
		}
		res := sn.EvalBatchEncrypted(Xbatch, Y, Xenc, weights, bias, Box, 10)
		corrects += res.Corrects
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}
