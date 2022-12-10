package cryptonet

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/cluster"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"runtime"
	"strconv"
	"testing"
	"time"
)
import "github.com/tuneinsight/lattigo/v3/ckks"
import "github.com/tuneinsight/lattigo/v3/rlwe"

var paramsLogN15, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ)
	LogP:         []int{50, 50, 50, 50},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     14,
	DefaultScale: float64(1 << 35),
})

var paramsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{60, 60},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

var paramsLogN14Mask, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{44, 44},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

//EXPERIMENT 1 - Model in clear - data encrypted:
//Querier sends encrypted data to server to get privacy preserving prediction
func TestCryptonet_EvalBatchEncrypted(t *testing.T) {
	//3892.666667ms for 41 batch with logn14
	//3516.000000ms for 1 sample with logn14 = -10% runtime

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	//First we load the network. Note that we pass a custom InitActivations method to the loader (have a look in cryptonet.go)
	loader := new(CNLoader)
	cn := loader.Load("cryptonet_packed.json", InitActivations)

	params := paramsLogN14
	features := 28 * 28
	rows, cols := cn.GetDimentions()

	//we define a new splitter providing all the information needed and we find how to split our model weights
	possibleSplits := cipherUtils.NewSplitter(1, features, rows, cols, params).FindSplits()
	possibleSplits.Print()

	//define a new Box: this is just a wrapper for all the cryptography related objects, like a toolbox
	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	splitInfo, _ := possibleSplits.ExctractInfo()
	batchSize := splitInfo.BatchSize

	//we have to set the batchSize this network accepts: the batch determines how the network is split
	cn.SetBatch(batchSize)

	// finally we define our network. Note that this network does not support bootstrapping (we don't need it)
	// and that is not encrypted (weights are in clear)
	cne := cn.NewHE(possibleSplits, false, false, 0, params.MaxLevel(), nil, poolSize, Box)
	fmt.Println("Encoded Cryptonet...")

	datacn := data.LoadData("cryptonet_data_nopad.json")
	err := datacn.Init(batchSize)
	utils.ThrowErr(err)

	result := utils.NewStats(batchSize)
	resultExp := utils.NewStats(batchSize)

	iters := 0
	maxIters := 15

	//generate and store rotation keys
	Box = cipherUtils.BoxWithRotations(Box, cne.GetRotations(Box.Params, nil), false, nil)

	for true {
		X, Y, err := datacn.Batch()
		Xbatch := plainUtils.NewDense(X)
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		//We create a new encrypted input passing the information from the splits
		Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
		utils.ThrowErr(err)
		//little optimization at client-side
		cipherUtils.PrepackBlocks(Xenc, splitInfo.ColsOfWeights[0], Box)

		if !debug {
			//Eval your network, easy as that!
			resHE, end := cne.Eval(Xenc)
			fmt.Println("End", end)
			resClear := cipherUtils.DecInput(resHE, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		} else {
			resHE, resExp, end := cne.EvalDebug(Xenc, Xbatch, cn, 1.0)
			fmt.Println("End", end)
			resClear := cipherUtils.DecInput(resHE, Box)
			utils.Predict(Y, 10, resClear)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
			corrects, accuracy, _ = utils.Predict(Y, 10, plainUtils.MatToArray(resExp))
			fmt.Println("Accuracy Expected: ", accuracy)
			resultExp.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		}
		iters++
	}
	result.PrintResult()
	if debug {
		fmt.Println()
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
}

//EXPERIMENT 2 - Model encrypted,data in clear:
//In this scenario the model is sent by server to the client in encrypted form
//Server offers an oblivious decryption service
//EDIT: this version uses localhost to simulate LAN environment
func TestCryptonet_EvalBatchClearModelEnc(t *testing.T) {

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	loader := new(CNLoader)
	cn := loader.Load("cryptonet_packed.json", InitActivations)

	params := paramsLogN14Mask
	features := 28 * 28
	rows, cols := cn.GetDimentions()
	possibleSplits := cipherUtils.NewSplitter(-1, features, rows, cols, params).FindSplits()

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	splitInfo, _ := possibleSplits.ExctractInfo()

	batchSize := splitInfo.BatchSize
	cn.SetBatch(batchSize)

	//note that this time the network is encrypted!
	cne := cn.NewHE(possibleSplits, true, false, 0, params.MaxLevel(), nil, poolSize, Box)
	fmt.Println("Encrypted Cryptonet...")

	datacn := data.LoadData("cryptonet_data_nopad.json")
	err := datacn.Init(batchSize)
	utils.ThrowErr(err)

	result := utils.NewStats(batchSize)
	resultExp := utils.NewStats(batchSize)

	iters := 0
	maxIters := 15

	//generate and store rotation keys
	Box = cipherUtils.BoxWithRotations(Box, cne.GetRotations(Box.Params, nil), false, nil)

	//start server with decryption oracle
	serverAddr := "127.0.0.1:8001"
	client, err := distributed.NewClient(serverAddr, Box, poolSize, true)
	utils.ThrowErr(err)
	server, err := distributed.NewServer(Box, serverAddr, true)
	go server.Listen()
	utils.ThrowErr(err)

	for true {
		X, Y, err := datacn.Batch()
		Xbatch := plainUtils.NewDense(X)
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		Xp, err := cipherUtils.NewPlainInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
		utils.ThrowErr(err)
		cipherUtils.PrepackBlocks(Xp, splitInfo.ColsOfWeights[0], Box)

		if !debug {
			start := time.Now()
			resHE, _ := cne.Eval(Xp)
			resMasked := client.StartProto(distributed.MASKING, resHE)
			end := time.Since(start)
			fmt.Println("End ", end)

			resClear := cipherUtils.DecodeInput(resMasked, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		} else {
			start := time.Now()
			resHE, resExp, _ := cne.EvalDebug(Xp, Xbatch, cn, 1.0)
			resMasked := client.StartProto(distributed.MASKING, resHE)
			end := time.Since(start)
			fmt.Println("End ", end)

			resClear := cipherUtils.DecodeInput(resMasked, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
			corrects, accuracy, _ = utils.Predict(Y, 10, plainUtils.MatToArray(resExp))
			fmt.Println("Accuracy Expected: ", accuracy)
			resultExp.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		}
		iters++
	}
	result.PrintResult()
	if debug {
		fmt.Println()
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
}

//EXPERIMENT 2 - Model encrypted,data in clear:
//In this scenario the model is sent by server to the client in encrypted form
//Server offers an oblivious decryption service
//This version spawns the server on a server on the iccluster
func TestCryptonet_EvalBatchClearModelEnc_LAN(t *testing.T) {

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	loader := new(CNLoader)
	cn := loader.Load("cryptonet_packed.json", InitActivations)

	params := paramsLogN14Mask
	features := 28 * 28
	rows, cols := cn.GetDimentions()
	possibleSplits := cipherUtils.NewSplitter(-1, features, rows, cols, params).FindSplits()

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	splitInfo, _ := possibleSplits.ExctractInfo()

	batchSize := splitInfo.BatchSize
	cn.SetBatch(batchSize)

	cne := cn.NewHE(possibleSplits, false, false, 0, params.MaxLevel(), nil, poolSize, Box)
	fmt.Println("Encoded Cryptonet...")

	datacn := data.LoadData("cryptonet_data_nopad.json")
	err := datacn.Init(batchSize)
	utils.ThrowErr(err)

	result := utils.NewStats(batchSize)
	resultExp := utils.NewStats(batchSize)

	iters := 0
	maxIters := 15

	//generate and store rotation keys
	Box = cipherUtils.BoxWithRotations(Box, cne.GetRotations(Box.Params, nil), false, nil)

	clusterConfig := cluster.ReadConfig("../cluster/config.json")
	serverAddr := clusterConfig.ClusterIps[1] //0 is the client

	client, err := distributed.NewClient(serverAddr+":"+strconv.Itoa(distributed.ServicePort), Box, poolSize, false)
	utils.ThrowErr(err)

	client.ClientSetup(serverAddr, Box.Sk)
	utils.ThrowErr(err)

	for true {
		X, Y, err := datacn.Batch()
		Xbatch := plainUtils.NewDense(X)
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		Xp, err := cipherUtils.NewPlainInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
		utils.ThrowErr(err)
		cipherUtils.PrepackBlocks(Xp, splitInfo.ColsOfWeights[0], Box)

		if !debug {
			start := time.Now()
			resHE, _ := cne.Eval(Xp)
			resMasked := client.StartProto(distributed.MASKING, resHE)
			end := time.Since(start)
			fmt.Println("End ", end)

			resClear := cipherUtils.DecodeInput(resMasked, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		} else {
			start := time.Now()
			resHE, resExp, _ := cne.EvalDebug(Xp, Xbatch, cn, 1.0)
			resMasked := client.StartProto(distributed.MASKING, resHE)
			end := time.Since(start)
			fmt.Println("End ", end)

			resClear := cipherUtils.DecodeInput(resMasked, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
			corrects, accuracy, _ = utils.Predict(Y, 10, plainUtils.MatToArray(resExp))
			fmt.Println("Accuracy Expected: ", accuracy)
			resultExp.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		}
		iters++
	}
	result.PrintResult()
	if debug {
		fmt.Println()
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
	client.StartProto(distributed.END, nil)
}
