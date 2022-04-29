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
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
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
	nn.Init(layers)

	batchSize := 128
	//for input block
	InRowP := 1
	InColP := 30 //inputs are 30x30
	nnb, _ := nn.NewBlockNN(batchSize, InRowP, InColP)

	// CRYPTO =========================================================================================================
	ckksParams := bootstrapping.DefaultParametersSparse[3].SchemeParams
	btpParams := bootstrapping.DefaultParametersSparse[3].BootstrappingParams
	btpCapacity := 2 //remaining levels
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	utils.ThrowErr(err)

	keyPath := fmt.Sprintf("/root/nn%d_paramSlots%d", layers, params.LogSlots())
	_, err = os.OpenFile(keyPath+"_sk", os.O_RDONLY, 0755)
	sk := new(rlwe.SecretKey)
	rtks := new(rlwe.RotationKeySet)
	kgen := ckks.NewKeyGenerator(params)
	if os.IsNotExist(err) {
		// create keys
		fmt.Println("Generating keys...")
		sk = kgen.GenSecretKey()
		inputInnerRows := batchSize / InRowP
		rotations := cipherUtils.GenRotations(inputInnerRows, len(nnb.Weights), nnb.InnerRows, nnb.InnerCols, params, &btpParams)
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

	//Encrypt weights in block form
	nne, _ := nnb.NewEncNN(batchSize, InRowP, btpCapacity, Box)

	//Load Dataset
	dataSn := data.LoadData("/root/nn_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	tot := 0
	iters := 0
	maxIters := 5
	var elapsed int64
	//Start Inference run
	fmt.Println("Starting inference on dataset...")
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil || iters == maxIters {
			//dataset completed
			break
		}
		X, _ := plainUtils.PartitionMatrix(plainUtils.NewDense(Xbatch), InRowP, InColP)
		Xenc, err := cipherUtils.NewEncInput(Xbatch, InRowP, InColP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		//correctsInBatch, duration := nn.EvalBatchEncrypted(X, Y, Xenc, weightsBlock, biasBlock, weightMatricesBlock, biasMatricesBlock, Box, 10)
		correctsInBatch, duration := EvalBatchEncrypted(nne, nnb, X, Y, Xenc, Box, 10, true)
		corrects += correctsInBatch
		elapsed += duration.Milliseconds()
		fmt.Println("Corrects/Tot:", correctsInBatch, "/", batchSize)
		tot += batchSize
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	fmt.Println("Latency (avg ms per batch):", float64(elapsed)/float64(iters))
}

func TestEvalDataEncModelEnc_Distributed(t *testing.T) {
	/*
		Setting:
			Querier has sk and pk
			Parties have collective pk' and sk' shares
			Querier sends data encrypted under pk' to cloud-cohort (parties)
			Inference is performed, and key-switched back to pk to be decrypted by querier
	*/
	layers := 20

	nn := LoadNN("/root/nn" + strconv.Itoa(layers) + "_packed.json")
	nn.Init(layers)

	batchSize := 128
	//for input block
	InRowP := 1
	InColP := 30 //inputs are 30x30
	nnb, _ := nn.NewBlockNN(batchSize, InRowP, InColP)

	// CRYPTO =========================================================================================================
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogQ:         []int{42, 40, 40, 40, 40, 40, 40}, //Log(PQ) <= 438 for LogN 14
		LogP:         []int{43, 43, 43},
		Sigma:        rlwe.DefaultSigma,
		LogSlots:     13,
		DefaultScale: float64(1 << 40),
	})
	// QUERIER
	kgen := ckks.NewKeyGenerator(params)
	skQ := kgen.GenSecretKey()
	encQ := ckks.NewEncryptor(params, skQ)
	decQ := ckks.NewDecryptor(params, skQ)

	BoxQ := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    nil,
		Decryptor:    decQ,
		Encryptor:    encQ,
		BootStrapper: nil,
	}
	// PARTIES
	// [!] All the keys for encryption, keySw, Relin can be produced by MPC protocols
	// [!] We assume that these protocols have been run in a setup phase by the parties
	prng, _ := lattigoUtils.NewKeyedPRNG([]byte{'t', 'e', 's', 't'})
	//Encrypt weights in block form
	nne, _ := nnb.NewEncNN(batchSize, InRowP, btpCapacity, Box)

}
