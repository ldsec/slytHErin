package nn

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"os"
	"strconv"
	"testing"
)

func TestEvalDataEncModelEnc(t *testing.T) {
	//data encrypted - model enc, not distributed
	/*
		5 runs measure:
		1 - 638s
		2 - 640s
		3 - 643s
		4 - 642s
		5 - 643s
		_____________
		5.009s/sample
	*/
	layers := 20

	nn := LoadNN("/root/nn" + strconv.Itoa(layers) + "_packed.json")

	nn.Init(layers)
	batchSize := 128
	//for input block
	rowP := 1
	colP := 30 //inputs are 30x30
	inputInnerRows := batchSize / rowP
	nnb, _ := nn.NewBlockNN(batchSize, rowP, colP)

	// CRYPTO
	ckksParams := bootstrapping.DefaultParametersSparse[3].SchemeParams
	btpParams := bootstrapping.DefaultParametersSparse[3].BootstrappingParams
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	utils.ThrowErr(err)

	// read or generate secret keys
	keyPath := fmt.Sprintf("/root/nn%d__paramSlots%d__batch%d", layers, params.LogSlots(), batchSize)
	_, err = os.OpenFile(keyPath+"_sk", os.O_RDONLY, 0755)
	sk := new(rlwe.SecretKey)
	rtks := new(rlwe.RotationKeySet)
	kgen := ckks.NewKeyGenerator(params)
	if os.IsNotExist(err) {
		// create keys
		fmt.Println("Generating keys...")
		sk = kgen.GenSecretKey()
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

	//load dataset
	dataSn := data.LoadData("/root/nn_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	//encrypt weights
	nne, _ := nnb.NewEncNN(batchSize, rowP, 2, Box, -1) //-1 means not distributed

	corrects := 0
	tot := 0
	iters := 0
	maxIters := 5
	var elapsed int64
	fmt.Println("Starting inference on dataset...")
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil && iters == maxIters {
			//dataset completed
			break
		}
		X, _ := plainUtils.PartitionMatrix(plainUtils.NewDense(Xbatch), rowP, colP)
		Xenc, err := cipherUtils.NewEncInput(Xbatch, rowP, colP, params.MaxLevel(), Box)
		utils.ThrowErr(err)
		correctsInBatch, duration := nne.EvalBatchEncrypted(nnb, X, Y, Xenc, Box, 10, true)
		corrects += correctsInBatch
		elapsed += duration.Milliseconds()
		fmt.Println("Corrects/Tot:", correctsInBatch, "/", batchSize)
		tot += batchSize
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	avg := float64(elapsed) / float64(iters)
	fmt.Println("Latency (avg ms per batch):", avg)
	fmt.Println("Latency (avg ms per sample):", avg/float64(batchSize))
}

func TestEvalDataEncModelEnc_Distributed(t *testing.T) {
	/*
		Setting:
			Querier has sk and pk
			Parties have collective pk' and sk' shares
			Querier sends data encrypted under pk' to cloud-cohort (parties)
			Inference is performed, and key-switched back to pk to be decrypted by querier

			The distributed setting corresponds to the Cloud-assisted setting of https://eprint.iacr.org/2020/304.pdf
			Root has one share of the key
			Topology is star with root at the centre, supposing small enough # of parties

		Runtime:
			Latency (avg ms per batch): 191676.8
			Latency (avg ms per sample): 1497.475
	*/
	layers := 20

	nn := LoadNN("/root/nn" + strconv.Itoa(layers) + "_packed.json")
	nn.Init(layers)

	batchSize := 64
	//for input block
	InRowP := 1
	InColP := 30 //inputs are 30x30
	inputInnerRows := batchSize / InRowP
	nnb, _ := nn.NewBlockNN(batchSize, InRowP, InColP)

	// CRYPTO =========================================================================================================
	//params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	//	LogN:         14,
	//	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	//	LogP:         []int{33, 33, 33},
	//	Sigma:        rlwe.DefaultSigma,
	//	LogSlots:     13,
	//	DefaultScale: float64(1 << 30),
	//})
	params, _ := ckks.NewParametersFromLiteral(ckks.DefaultParams[2])
	// QUERIER
	kgenQ := ckks.NewKeyGenerator(params)
	skQ := kgenQ.GenSecretKey()
	pkQ := kgenQ.GenPublicKey(skQ)
	decQ := ckks.NewDecryptor(params, skQ)

	// PARTIES
	// [!] All the keys for encryption, keySw, Relin can be produced by MPC protocols
	// [!] We assume that these protocols have been run in a setup phase by the parties

	parties := 3
	crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})
	skShares, skP, pkP, kgenP := distributed.DummyEncKeyGen(params, crs, parties)
	rlk := distributed.DummyRelinKeyGen(params, crs, skShares)
	rotations := cipherUtils.GenRotations(inputInnerRows, len(nnb.Weights), nnb.InnerRows, nnb.InnerCols, params, nil)
	rtks := kgenP.GenRotationKeysForRotations(rotations, true, skP)
	decP := ckks.NewDecryptor(params, skP)

	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),                                             //public
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}), //from parties
		Decryptor:    decP,                                                                //from parties for debug
		Encryptor:    ckks.NewEncryptor(params, pkP),                                      //from parties
		BootStrapper: nil,
	}

	//Create distributed parties
	master, err := distributed.NewDummyMaster(skShares[0], pkP, params, parties)
	utils.ThrowErr(err)
	players := make([]*distributed.DummyPlayer, parties-1)
	for i := 0; i < parties-1; i++ {
		players[i], err = distributed.NewDummyPlayer(skShares[i+1], pkP, params, i+1, master.P2MChans[i+1], master.M2PChans[i+1])
		utils.ThrowErr(err)
	}
	//start master
	master.Listen()
	//start players
	for _, p := range players {
		go p.Dispatch()
	}

	// info for bootstrapping
	var minLevel int
	var ok bool
	if minLevel, _, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q()); ok != true || minLevel+1 > params.MaxLevel() {
		utils.ThrowErr(errors.New("Not enough levels to ensure correcness and 128 security"))
	}
	//Encrypt weights in block form
	nne, _ := nnb.NewEncNN(batchSize, InRowP, params.MaxLevel(), Box, minLevel)

	//Load Dataset
	dataSn := data.LoadData("/root/nn_data.json")
	err = dataSn.Init(batchSize)
	if err != nil {
		fmt.Println(err)
		return
	}

	corrects := 0
	correctsPlain := 0
	tot := 0
	iters := 0
	maxIters := 10
	debug := true
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
		correctsInBatchPlain, correctsInBatch, duration := EvalBatchEncryptedDistributed(nne, nnb, X, Y, Xenc, Box, pkQ, decQ, minLevel, 10, debug, master, players)
		corrects += correctsInBatch
		correctsPlain += correctsInBatchPlain
		elapsed += duration.Milliseconds()
		fmt.Println("Corrects/Tot:", correctsInBatch, "/", batchSize)
		tot += batchSize
		iters++
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
	if debug {
		fmt.Println("Expected Accuracy:", float64(correctsPlain)/float64(tot))
	}

	avg := float64(elapsed) / float64(iters)
	fmt.Println("Latency (avg ms per batch):", avg)
	fmt.Println("Latency (avg ms per sample):", avg/float64(batchSize))
}
