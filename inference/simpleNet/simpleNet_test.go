package simpleNet

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestEvalPlain(t *testing.T) {
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.Init()

	batchSize := 8
	inputLayerDim, _ := utils.BuildKernelMatrix(sn.Conv1.Weight).Dims()
	dataSn := data.LoadData("../../training/data/simpleNet_data.json")
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
	sn.Init()

	batchSize := 128
	dataSn := data.LoadData("../../training/data/simpleNet_data.json")
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
	//data encrypted - model in clear

	//local run
	sn := LoadSimpleNet("simpleNet_packed.json")
	sn.Init()

	batchSize := 64
	//for input block
	rowP := 1
	colP := 29

	conv1M := utils.BuildKernelMatrix(sn.Conv1.Weight)
	conv1MB, _ := plainUtils.PartitionMatrix(conv1M, colP, 13*5)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := utils.BuildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)

	pool1M := utils.BuildKernelMatrix(sn.Pool1.Weight)
	pool1MB, _ := plainUtils.PartitionMatrix(pool1M, 13*5, 10)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := utils.BuildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)

	pool2M := utils.BuildKernelMatrix(sn.Pool2.Weight)
	pool2MB, err := plainUtils.PartitionMatrix(pool2M, 10, 1)
	utils.ThrowErr(err)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := utils.BuildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	//crypto
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{60, 57, 57, 57, 57, 57, 57, 57}, //9 x 60 --> Log(Q) <= 881 for LogN 15
		LogP:         []int{61, 61, 61},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 57),
	})
	//ckksParams := bootstrapping.DefaultCKKSParameters[4]
	//btpParams := bootstrapping.DefaultParameters[4]
	//params, err := ckks.NewParametersFromLiteral(ckksParams)

	utils.ThrowErr(err)
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
	//rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, &btpParams)
	rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, nil)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	//btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	utils.ThrowErr(err)
	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
	}

	weightsBlock := make([]*cipherUtils.PlainWeightDiag, 3)
	biasBlock := make([]*cipherUtils.PlainInput, 3)

	//plaintext weights and bias
	weightsBlock[0], _ = cipherUtils.NewPlainWeightDiag(plainUtils.MatToArray(weightMatrices[0]),
		colP, 13*5, batchSize, params.MaxLevel(), Box)
	weightsBlock[1], _ = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(plainUtils.MulByConst(weightMatrices[1], 1.0/sn.ReLUApprox.Interval)),
		13*5, 10, batchSize, params.MaxLevel()-1, Box)
	weightsBlock[2], err = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(plainUtils.MulByConst(weightMatrices[2], 1.0/sn.ReLUApprox.Interval)),
		10, 1, batchSize, params.MaxLevel()-1-1-2, Box)
	utils.ThrowErr(err)

	biasBlock[0], _ = cipherUtils.NewPlainInput(plainUtils.MatToArray(biasMatrices[0]),
		rowP, 13*5, params.MaxLevel()-1, Box)
	biasBlock[1], _ = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(plainUtils.MulByConst(biasMatrices[1], 1.0/sn.ReLUApprox.Interval)),
		rowP, 10, params.MaxLevel()-1-1, Box)
	biasBlock[2], err = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(plainUtils.MulByConst(biasMatrices[2], 1.0/sn.ReLUApprox.Interval)),
		rowP, 1, params.MaxLevel()-1-1-2-1, Box)
	utils.ThrowErr(err)
	fmt.Println("Created block matrixes...")

	dataSn := data.LoadData("simpleNet_data.json")
	//dataSn := data.LoadSimpleNetData("/root/simpleNet_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil {
			//dataset completed
			break
		}
		Xenc, err := cipherUtils.NewEncInput(Xbatch, rowP, colP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		res := sn.EvalBatchEncrypted_Debug(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10)
		fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
		corrects += res.Corrects
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}

func TestEvalDataEncModelClearCompressed(t *testing.T) {
	/*
		data encrypted - model in clear
		model is optimized by compressing conv and pool1 in 1 layer
		~4.8s per 70 batch ==> 14,58 im/s
	*/
	sn := LoadSimpleNet("/francesco/simpleNet_packed.json")
	sn.Init()
	//crypto
	//ckksParams := ckks.ParametersLiteral{
	//	LogN:         15,
	//	LogQ:         []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 438 for LogN 14
	//	LogP:         []int{38, 38, 38, 38},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     14,
	//	DefaultScale: float64(1 << 26),
	//}

	//ckksParams := ckks.ParametersLiteral{
	//	LogN:         14,
	//	LogQ:         []int{30, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 438 for LogN 14
	//	LogP:         []int{38, 38, 38, 38},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     13,
	//	DefaultScale: float64(1 << 26),
	//}

	ckksParams := ckks.ParametersLiteral{
		LogN:         13,
		LogQ:         []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 218 for LogN 3
		LogP:         []int{33},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     12,
		DefaultScale: float64(1 << 26),
	}

	params, err := ckks.NewParametersFromLiteral(ckksParams)
	rowP := 1
	colP := 29
	batchSize := rowP * cipherUtils.GetOptimalInnerRows(29, params)
	fmt.Println("Batch: ", batchSize)
	//for input block

	conv1M := utils.BuildKernelMatrix(sn.Conv1.Weight)
	conv1MB, _ := plainUtils.PartitionMatrix(conv1M, colP, 13*5)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := utils.BuildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)

	pool1M := utils.BuildKernelMatrix(sn.Pool1.Weight)
	pool1MB, _ := plainUtils.PartitionMatrix(pool1M, 13*5, 10)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := utils.BuildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)

	pool2M := utils.BuildKernelMatrix(sn.Pool2.Weight)
	pool2MB, err := plainUtils.PartitionMatrix(pool2M, 10, 1)
	utils.ThrowErr(err)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := utils.BuildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	utils.ThrowErr(err)
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
	//rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, &btpParams)
	rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, nil)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	//btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	utils.ThrowErr(err)
	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
	}

	weightsBlock := make([]*cipherUtils.PlainWeightDiag, 3)
	biasBlock := make([]*cipherUtils.PlainInput, 3)

	//plaintext weights and bias
	var compressed mat.Dense
	compressed.Mul(conv1M, pool1M)
	var biasCompressed mat.Dense
	biasCompressed.Mul(bias1M, pool1M)
	biasCompressed.Add(&biasCompressed, bias2M)

	w0 := plainUtils.MulByConst(&compressed, 1.0/sn.ReLUApprox.Interval)
	weightsBlock[0], _ = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(w0),
		colP, 10, batchSize, params.MaxLevel(), Box)

	w1 := plainUtils.MulByConst(weightMatrices[2], 1.0/sn.ReLUApprox.Interval)
	weightsBlock[1], err = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(w1),
		10, 1, batchSize, params.MaxLevel()-1-2, Box)
	utils.ThrowErr(err)

	b0 := plainUtils.MulByConst(&biasCompressed, 1.0/sn.ReLUApprox.Interval)
	biasBlock[0], _ = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(b0),
		rowP, 10, params.MaxLevel()-1, Box)
	b1 := plainUtils.MulByConst(biasMatrices[2], 1.0/sn.ReLUApprox.Interval)
	biasBlock[1], err = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(b1),
		rowP, 1, params.MaxLevel()-1-2-1, Box)
	utils.ThrowErr(err)
	fmt.Println("Created block matrixes...")

	dataSn := data.LoadData("simpleNet_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	iters := 0
	maxIters := 10
	var elapsed int64
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		Xenc, err := cipherUtils.NewEncInput(Xbatch, rowP, colP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		//res := sn.EvalBatchEncryptedCompressed(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10, false)
		res := sn.EvalBatchEncryptedCompressed_Light(Y, Xenc, weightsBlock, biasBlock, Box, 10)
		fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
		corrects += res.Corrects
		tot += batchSize
		elapsed += res.Time.Milliseconds()
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
}

func TestEvalDataEncModelClearCompressed_withActivators(t *testing.T) {
	/*
		data encrypted - model in clear
		model is optimized by compressing conv and pool1 in 1 layer
		~4.8s per 70 batch ==> 14,58 im/s
	*/
	sn := LoadSimpleNet("/francesco/simpleNet_packed.json")
	sn.Init()
	//crypto
	ckksParams := ckks.ParametersLiteral{
		LogN:         15,
		LogQ:         []int{33, 26, 26, 26, 26, 26, 26}, //Log(PQ)
		LogP:         []int{50, 50, 50, 50},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     14,
		DefaultScale: float64(1 << 26),
	}

	//ckksParams := ckks.ParametersLiteral{
	//	LogN:         14,
	//	LogQ:         []int{30, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 438 for LogN 14
	//	LogP:         []int{38, 38, 38, 38},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     13,
	//	DefaultScale: float64(1 << 26),
	//}

	//ckksParams := ckks.ParametersLiteral{
	//	LogN:         13,
	//	LogQ:         []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 218 for LogN 13
	//	LogP:         []int{33},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     12,
	//	DefaultScale: float64(1 << 26),
	//}

	params, err := ckks.NewParametersFromLiteral(ckksParams)
	rowP := 10
	colP := 29
	batchSize := rowP * cipherUtils.GetOptimalInnerRows(29, params)
	fmt.Println("Batch: ", batchSize)
	//for input block

	conv1M := utils.BuildKernelMatrix(sn.Conv1.Weight)
	conv1MB, _ := plainUtils.PartitionMatrix(conv1M, colP, 13*5)
	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := utils.BuildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)

	pool1M := utils.BuildKernelMatrix(sn.Pool1.Weight)
	pool1MB, _ := plainUtils.PartitionMatrix(pool1M, 13*5, 10)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := utils.BuildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)

	pool2M := utils.BuildKernelMatrix(sn.Pool2.Weight)
	pool2MB, err := plainUtils.PartitionMatrix(pool2M, 10, 1)
	utils.ThrowErr(err)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := utils.BuildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	utils.ThrowErr(err)
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
	//rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, &btpParams)
	rotations := cipherUtils.GenRotations(inputInnerRows, len(weightMatricesBlock), rowsW, colsW, params, nil)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	//btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	utils.ThrowErr(err)
	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
	}

	weightsBlock := make([]*cipherUtils.PlainWeightDiag, 2)
	biasBlock := make([]*cipherUtils.PlainInput, 2)
	activators := make([]*cipherUtils.Activator, 2)

	//plaintext weights and bias
	var compressed mat.Dense
	compressed.Mul(conv1M, pool1M)
	var biasCompressed mat.Dense
	biasCompressed.Mul(bias1M, pool1M)
	biasCompressed.Add(&biasCompressed, bias2M)

	w0 := plainUtils.MulByConst(&compressed, 1.0/sn.ReLUApprox.Interval)
	weightsBlock[0], _ = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(w0),
		colP, 10, inputInnerRows, params.MaxLevel(), Box)
	activators[0], err = cipherUtils.NewActivator(sn.ReLUApprox, params.MaxLevel()-1, params.DefaultScale(), batchSize/rowP, weightsBlock[0].InnerCols, rowP, weightsBlock[0].ColP, Box)
	utils.ThrowErr(err)
	w1 := plainUtils.MulByConst(weightMatrices[2], 1.0/sn.ReLUApprox.Interval)
	weightsBlock[1], err = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(w1),
		10, 1, inputInnerRows, params.MaxLevel()-1-2, Box)
	activators[1], err = cipherUtils.NewActivator(sn.ReLUApprox, params.MaxLevel()-1-2-1, params.DefaultScale(), batchSize/rowP, weightsBlock[1].InnerCols, rowP, weightsBlock[1].ColP, Box)
	utils.ThrowErr(err)
	utils.ThrowErr(err)

	b0 := plainUtils.MulByConst(&biasCompressed, 1.0/sn.ReLUApprox.Interval)
	biasBlock[0], _ = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(b0),
		rowP, 10, params.MaxLevel()-1, Box)
	b1 := plainUtils.MulByConst(biasMatrices[2], 1.0/sn.ReLUApprox.Interval)
	biasBlock[1], err = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(b1),
		rowP, 1, params.MaxLevel()-1-2-1, Box)
	utils.ThrowErr(err)
	fmt.Println("Created block matrixes...")

	dataSn := data.LoadData("simpleNet_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	iters := 0
	maxIters := 10
	var elapsed int64
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		Xenc, err := cipherUtils.NewEncInput(Xbatch, rowP, colP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		//res := sn.EvalBatchEncryptedCompressed(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10, false)
		res := sn.EvalBatchEncryptedCompressed_withActivator(Y, Xenc, weightsBlock, biasBlock, activators, Box, 10)
		fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
		corrects += res.Corrects
		tot += batchSize
		elapsed += res.Time.Milliseconds()
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
}

func TestEvalDataEncModelClearCompressed_withActivators_NoPad(t *testing.T) {
	/*
		data encrypted - model in clear (here data are not padded)
		model is optimized by compressing conv and pool1 in 1 layer
		~4.8s per 70 batch ==> 14,58 im/s
	*/
	sn := LoadSimpleNet("/francesco/simplenet_nopad.json")
	sn.Init()
	//crypto
	//ckksParams := ckks.ParametersLiteral{
	//	LogN:         15,
	//	LogQ:         []int{33, 26, 26, 26, 26, 26, 26}, //Log(PQ)
	//	LogP:         []int{50, 50, 50, 50},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     14,
	//	DefaultScale: float64(1 << 26),
	//}

	ckksParams := ckks.ParametersLiteral{
		LogN:         13,
		LogQ:         []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 218 for LogN 13
		LogP:         []int{33},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     12,
		DefaultScale: float64(1 << 26),
	}

	params, err := ckks.NewParametersFromLiteral(ckksParams)
	rowP := 1
	colP := 28 //49 1, 28 2
	colPConv := 2
	//for input block
	conv1M := utils.BuildKernelMatrix(sn.Conv1.Weight)
	conv1MB, _ := plainUtils.PartitionMatrix(conv1M, colP, colPConv)

	batchSize := rowP * cipherUtils.GetOptimalInnerRows(plainUtils.Max(784/colP, conv1MB.InnerCols), params) //here the cols should be the biggest in the pipeline btw
	inputInnerRows := batchSize / rowP
	fmt.Println("Batch: ", batchSize)

	inputLayerDim := plainUtils.NumCols(conv1M)
	bias1M := utils.BuildBiasMatrix(sn.Conv1.Bias, inputLayerDim, batchSize)
	pool1M := utils.BuildKernelMatrix(sn.Pool1.Weight)
	inputLayerDim = plainUtils.NumCols(pool1M)
	bias2M := utils.BuildBiasMatrix(sn.Pool1.Bias, inputLayerDim, batchSize)
	pool2M := utils.BuildKernelMatrix(sn.Pool2.Weight)

	utils.ThrowErr(err)
	inputLayerDim = plainUtils.NumCols(pool2M)
	bias3M := utils.BuildBiasMatrix(sn.Pool2.Bias, inputLayerDim, batchSize)

	weightMatrices := []*mat.Dense{conv1M, pool1M, pool2M}
	biasMatrices := []*mat.Dense{bias1M, bias2M, bias3M}

	utils.ThrowErr(err)
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	//init rotations
	//rotations are performed between submatrixes

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	//btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	utils.ThrowErr(err)
	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    nil,
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
	}

	weightsBlock := make([]*cipherUtils.PlainWeightDiag, 2)
	biasBlock := make([]*cipherUtils.PlainInput, 2)
	activators := make([]*cipherUtils.Activator, 2)

	//plaintext weights and bias
	var compressed mat.Dense
	compressed.Mul(conv1M, pool1M)
	var biasCompressed mat.Dense
	biasCompressed.Mul(bias1M, pool1M)
	biasCompressed.Add(&biasCompressed, bias2M)

	w0 := plainUtils.MulByConst(&compressed, 1.0/sn.ReLUApprox.Interval)
	weightsBlock[0], _ = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(w0),
		colP, colPConv, inputInnerRows, params.MaxLevel(), Box)
	activators[0], err = cipherUtils.NewActivator(sn.ReLUApprox, params.MaxLevel()-1, params.DefaultScale(), batchSize/rowP, weightsBlock[0].InnerCols, rowP, weightsBlock[0].ColP, Box)
	utils.ThrowErr(err)
	w1 := plainUtils.MulByConst(weightMatrices[2], 1.0/sn.ReLUApprox.Interval)
	weightsBlock[1], err = cipherUtils.NewPlainWeightDiag(
		plainUtils.MatToArray(w1),
		colPConv, 1, inputInnerRows, params.MaxLevel()-1-2, Box)
	activators[1], err = cipherUtils.NewActivator(sn.ReLUApprox, params.MaxLevel()-1-2-1, params.DefaultScale(), batchSize/rowP, weightsBlock[1].InnerCols, rowP, weightsBlock[1].ColP, Box)
	utils.ThrowErr(err)

	b0 := plainUtils.MulByConst(&biasCompressed, 1.0/sn.ReLUApprox.Interval)
	biasBlock[0], _ = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(b0),
		rowP, colPConv, params.MaxLevel()-1, Box)
	b1 := plainUtils.MulByConst(biasMatrices[2], 1.0/sn.ReLUApprox.Interval)
	biasBlock[1], err = cipherUtils.NewPlainInput(
		plainUtils.MatToArray(b1),
		rowP, 1, params.MaxLevel()-1-2-1, Box)
	utils.ThrowErr(err)

	rotations := cipherUtils.GenRotations(inputInnerRows, len(weightsBlock), []int{weightsBlock[0].InnerRows, weightsBlock[1].InnerRows}, []int{weightsBlock[0].InnerCols, weightsBlock[1].InnerCols}, params, nil)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	Box.Evaluator = ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: kgen.GenRelinearizationKey(sk, 2), Rtks: rtks})
	activators[0].Box = Box
	activators[1].Box = Box
	fmt.Println("Created block matrixes...")

	dataSn := data.LoadData("simpleNet_data_nopad.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	iters := 0
	maxIters := 10
	var elapsed int64
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		Xenc, err := cipherUtils.NewEncInput(Xbatch, rowP, colP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		//res := sn.EvalBatchEncryptedCompressed(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10, false)
		res := sn.EvalBatchEncryptedCompressed_withActivator(Y, Xenc, weightsBlock, biasBlock, activators, Box, 10)
		fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
		corrects += res.Corrects
		tot += batchSize
		elapsed += res.Time.Milliseconds()
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
}
