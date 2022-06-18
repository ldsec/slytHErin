package simpleNet

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/data"
	md "github.com/ldsec/dnn-inference/inference/multidim"
	pu "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	"math"
	"testing"
)

//Testing SimpleNet with MultiDimentional packing for enhanced throughput

func Test_BatchEncrypted(t *testing.T) {
	debug := true
	sn := LoadSimpleNet("simplenet_packed.json")
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

	//8 levels neeeded
	ckksParams := ckks2.ParametersLiteral{
		LogN:     14,
		LogQ:     []int{45, 35, 35, 35, 35, 35, 35, 35, 35}, //Log(PQ) <= 438 for LogN 14
		LogP:     []int{37, 37, 37},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 13,
		Scale:    float64(1 << 35),
	}

	params, err := ckks2.NewParametersFromLiteral(ckksParams)

	features := 784 //MNIST
	batchSize := 256
	innerDim := int(math.Ceil(float64(params.N()) / (2.0 * float64(batchSize))))
	fmt.Printf("Input Dense: Rows %d, Cols %d --> InnerDim: %d\n", batchSize, features, innerDim)
	dataSn := data.LoadData("simpleNet_data_nopad.json")
	err = dataSn.Init(batchSize)
	utils.ThrowErr(err)
	X, Y, err := dataSn.Batch()
	Xpacked := PackBatchParallel(pu.NewDense(X), innerDim, params)

	weightMatrices, biasMatrices := sn.CompressLayers(sn.BuildParams(batchSize))
	weightMatricesRescaled, biasMatricesRescaled := sn.RescaleForActivation(weightMatrices, biasMatrices)
	kgen := ckks2.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rlk := kgen.GenRelinearizationKey(sk, 2)

	//init rotations
	//rotations are performed between submatrixes

	enc := ckks2.NewEncryptor(params, sk)
	dec := ckks2.NewDecryptor(params, sk)
	//btp, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})
	utils.ThrowErr(err)

	Box := md.Ckks2Box{
		Params:       params,
		Encoder:      ckks2.NewEncoder(params),
		Evaluator:    nil,
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
	}

	//convert to multidim packing
	snMD := sn.ConvertToMDPack(Xpacked.Batches(), innerDim, Xpacked.Rows(), Xpacked.Cols(), weightMatricesRescaled, biasMatricesRescaled, params, Box.Encoder)
	//generate rotations and rot keys
	rotations := snMD.GenerateRotations(params)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	Box.Evaluator = ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rtks})

	snMD.InitPmMultiplier(params, Box.Evaluator)
	batchEnc := md.NewBatchEncryptor(params, sk)

	corrects := 0
	tot := 0
	iters := 0
	maxIters := int(math.Ceil(float64(1024 / batchSize)))
	var elapsed int64

	for true {
		Xenc := batchEnc.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Xpacked)
		fmt.Printf("Input dimentions:\nRows:%d\nCols:%d\nInnerDim:%d\nBatches:%d\n\n", Xenc.Rows(), Xenc.Cols(), Xenc.Dim(), Xpacked.Batches())
		var res SimpleNetPipeLine
		if !debug {
			res = snMD.EvalBatchEncrypted(Y, Xenc, Box)
		} else {
			res = snMD.EvalBatchEncrypted_Debug(Y, Xenc, Box, pu.NewDense(X), weightMatrices, biasMatrices)
		}
		fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
		corrects += res.Corrects
		tot += batchSize
		elapsed += res.Time.Milliseconds()
		iters++
		X, Y, err = dataSn.Batch()
		Xpacked = PackBatchParallel(pu.NewDense(X), innerDim, params)
		if err != nil || iters >= maxIters {
			break
		}

	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
}
