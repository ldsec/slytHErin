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
	//local run
	//sn := LoadSimpleNet("../../training/models/simpleNet.json")
	//cluster run
	sn := LoadSimpleNet("/root/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 128
	//for input block
	rowP := 1
	colP := 29

	conv1M := buildKernelMatrix(sn.Conv1.Weight)
	conv1MB, _ := plainUtils.PartitionMatrix(conv1M, colP, 13*5)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := buildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)

	pool1M := buildKernelMatrix(sn.Pool1.Weight)
	pool1MB, _ := plainUtils.PartitionMatrix(pool1M, 13*5, 10)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := buildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)

	pool2M := buildKernelMatrix(sn.Pool2.Weight)
	pool2MB, _ := plainUtils.PartitionMatrix(pool2M, 10, 1)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := buildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	//crypto
	ckksParams := bootstrapping.DefaultCKKSParameters[4]
	btpParams := bootstrapping.DefaultParameters[4]
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	if err != nil {
		panic(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	//init rotations
	weightMatricesBlock := []*plainUtils.BMatrix{conv1MB, pool1MB, pool2MB}
	rowsW := make([]int, len(weightMatricesBlock))
	colsW := make([]int, len(weightMatricesBlock))
	for w := range weightMatricesBlock {
		rowsW[w], colsW[w] = weightMatricesBlock[w].InnerRows, weightMatricesBlock[w].InnerCols
	}
	//rotations are performed between submatrixes
	inputInnerRows := batchSize / rowP
	rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, &btpParams)
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

	//build weight and bias
	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	weightsBlock := make([]*cipherUtils.PlainWeightDiag, 3)
	biasBlock := make([]*cipherUtils.PlainInput, 3)

	weightsBlock[0], _ = cipherUtils.NewPlainWeightDiag(plainUtils.MatToArray(weightMatrices[0]),
		colP, 13*5, batchSize, Box)
	weightsBlock[1], _ = cipherUtils.NewPlainWeightDiag(plainUtils.MatToArray(plainUtils.MulByConst(weightMatrices[1], 1.0/10.0)),
		13*5, 10, batchSize, Box)
	weightsBlock[2], _ = cipherUtils.NewPlainWeightDiag(plainUtils.MatToArray(plainUtils.MulByConst(weightMatrices[2], 1.0/10.0)),
		10, 1, batchSize, Box)
	biasBlock[0], _ = cipherUtils.NewPlainInput(plainUtils.MatToArray(biasMatrices[0]),
		rowP, 13*5, Box)
	biasBlock[1], _ = cipherUtils.NewPlainInput(plainUtils.MatToArray(plainUtils.MulByConst(biasMatrices[1], 1.0/10.0)),
		rowP, 10, Box)
	biasBlock[2], _ = cipherUtils.NewPlainInput(plainUtils.MatToArray(plainUtils.MulByConst(biasMatrices[2], 1.0/10.0)),
		rowP, 1, Box)
	fmt.Println("Created block matrixes...")

	//dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	dataSn := data.LoadSimpleNetData("/root/simpleNet_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		Xenc, _ := cipherUtils.NewEncInput(Xbatch, rowP, colP, Box)
		if err != nil {
			//dataset completed
			break
		}
		res := sn.EvalBatchEncrypted(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10)
		corrects += res.Corrects
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}
