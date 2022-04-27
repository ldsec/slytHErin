package nn

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
	"os"
	"strconv"
	"testing"
)

func TestEvalDataEncModelEnc(t *testing.T) {
	//data encrypted - model enc, not distributed
	/*
		Current time:
		NN20 = 662s per 128 batch -> 5.17s per sample
	*/
	layers := 20

	nn := LoadNN("/root/nn" + strconv.Itoa(layers) + "_packed.json")

	nn.Init()
	batchSize := 128
	//for input block
	rowP := 1
	colP := 30 //inputs are 30x30

	convM := utils.BuildKernelMatrix(nn.Conv.Weight)
	convMB, _ := plainUtils.PartitionMatrix(convM, colP, 28) //900x840 --> submatrices are 30x30
	inputLayerDim := plainUtils.NumCols(convM)
	biasConvM := utils.BuildBiasMatrix(nn.Conv.Bias, inputLayerDim, batchSize)
	biasConvMB, _ := plainUtils.PartitionMatrix(biasConvM, rowP, convMB.ColP)

	denseMatrices := make([]*mat.Dense, layers)
	denseBiasMatrices := make([]*mat.Dense, layers)
	denseMatricesBlock := make([]*plainUtils.BMatrix, layers)
	denseBiasMatricesBlock := make([]*plainUtils.BMatrix, layers)

	for i := 0; i < layers; i++ {
		denseMatrices[i] = utils.BuildKernelMatrix(nn.Dense[i].Weight)
		inputLayerDim = plainUtils.NumCols(denseMatrices[i])
		denseBiasMatrices[i] = utils.BuildBiasMatrix(nn.Dense[i].Bias, inputLayerDim, batchSize)
		if i == 0 {
			//840x92 --> 30x23
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 28, 4)
		} else if i == layers-1 {
			//92x10 --> 23x10
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 4, 1)
		} else {
			//92x92 --> 23x23
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 4, 4)
		}
		denseBiasMatricesBlock[i], _ = plainUtils.PartitionMatrix(
			denseBiasMatrices[i],
			rowP,
			denseMatricesBlock[i].ColP)
	}

	weightMatrices := []*mat.Dense{convM}
	weightMatrices = append(weightMatrices, denseMatrices...)
	biasMatrices := []*mat.Dense{biasConvM}
	biasMatrices = append(biasMatrices, denseBiasMatrices...)

	weightMatricesBlock := []*plainUtils.BMatrix{convMB}
	weightMatricesBlock = append(weightMatricesBlock, denseMatricesBlock...)
	biasMatricesBlock := []*plainUtils.BMatrix{biasConvMB}
	biasMatricesBlock = append(biasMatricesBlock, denseBiasMatricesBlock...)
	rowsW := make([]int, len(weightMatricesBlock))
	colsW := make([]int, len(weightMatricesBlock))
	rowsPW := make([]int, len(weightMatricesBlock))
	colsPW := make([]int, len(weightMatricesBlock))
	for w := range weightMatricesBlock {
		rowsW[w], colsW[w] = weightMatricesBlock[w].InnerRows, weightMatricesBlock[w].InnerCols
		rowsPW[w], colsPW[w] = weightMatricesBlock[w].RowP, weightMatricesBlock[w].ColP
	}
	// ======= CRYPTO =======
	ckksParams := bootstrapping.DefaultParametersSparse[3].SchemeParams
	btpParams := bootstrapping.DefaultParametersSparse[3].BootstrappingParams
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	utils.ThrowErr(err)

	keyPath := fmt.Sprintf("/root/nn%d", layers)
	_, err = os.OpenFile(keyPath+"_sk", os.O_RDONLY, 0755)
	sk := new(rlwe.SecretKey)
	rtks := new(rlwe.RotationKeySet)
	kgen := ckks.NewKeyGenerator(params)
	if os.IsNotExist(err) {
		// create keys
		sk = kgen.GenSecretKey()
		inputInnerRows := batchSize / rowP
		rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, &btpParams)
		fmt.Println("Generating rot keys...")
		rtks = kgen.GenRotationKeysForRotations(rotations, true, sk)
		cipherUtils.SerializeKeys(keyPath, sk, rtks)
	} else {
		//read keys
		sk, rtks = cipherUtils.DesereliazeKeys(keyPath)
	}
	rlk := kgen.GenRelinearizationKey(sk, 2)
	fmt.Println("Done")
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	evk := bootstrapping.GenEvaluationKeys(btpParams, params, sk)
	btp, err := bootstrapping.NewBootstrapper(params, btpParams, evk)
	utils.ThrowErr(err)
	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: btp,
	}

	weightsBlock := make([]*cipherUtils.EncWeightDiag, layers+1)
	biasBlock := make([]*cipherUtils.EncInput, layers+1)

	level := params.MaxLevel()
	fmt.Println("Creating weights encrypted block matrices...")
	for i := 0; i < layers+1; i++ {
		fmt.Println("Sizes:")
		fmt.Printf("RowP: %d, ColP: %d, InnerR: %d, InnerC: %d \n", rowsPW[i], colsPW[i], rowsW[i], colsW[i])
		weightsBlock[i], _ = cipherUtils.NewEncWeightDiag(
			plainUtils.MatToArray(plainUtils.MulByConst(weightMatrices[i], 1.0/nn.ReLUApprox.Interval)),
			rowsPW[i], colsPW[i], batchSize, level, Box)
		level--
		biasBlock[i], _ = cipherUtils.NewEncInput(
			plainUtils.MatToArray(plainUtils.MulByConst(biasMatrices[i], 1.0/nn.ReLUApprox.Interval)),
			rowP, colsPW[i], level, Box)
		level -= 2 //activation
		if level <= 0 {
			//lvl after btp is 2 --> #Qs after STC
			level = 2
		}
	}
	fmt.Println("Done...")

	dataSn := data.LoadData("/root/nn_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	iters := 0
	var elapsed int64
	fmt.Println("Starting inference on dataset...")
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil {
			//dataset completed
			break
		}
		X, _ := plainUtils.PartitionMatrix(plainUtils.NewDense(Xbatch), rowP, colP)
		Xenc, err := cipherUtils.NewEncInput(Xbatch, rowP, colP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		correctsInBatch, duration := nn.EvalBatchEncrypted(X, Y, Xenc, weightsBlock, biasBlock, weightMatricesBlock, biasMatricesBlock, Box, 10)
		corrects += correctsInBatch
		elapsed += duration.Milliseconds()
		fmt.Println("Corrects/Tot:", correctsInBatch, "/", batchSize)
		tot += batchSize
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
}
