package cryptonet

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/cluster"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"runtime"
	"strconv"
	"testing"
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
	//5.6 for 41 btach with logn14

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	loader := new(CNLoader)
	cn := loader.Load("cryptonet_packed.json")

	params := paramsLogN14
	features := 28 * 28
	rows, cols := cn.GetDimentions()
	possibleSplits := cipherUtils.FindSplits(-1, features, rows, cols, params)

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}

	splits := possibleSplits[0]
	cipherUtils.PrintSetOfSplits(splits)

	splitInfo, _ := cipherUtils.ExctractInfo(splits)

	batchSize := splitInfo.InputRows * splitInfo.InputRowP
	cn.SetBatch(batchSize)

	Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

	cne := cn.NewHE(splits, false, false, 0, params.MaxLevel(), nil, poolSize, Box)
	fmt.Println("Encoded Cryptonet...")

	datacn := data.LoadData("cryptonet_data_nopad.json")
	err := datacn.Init(batchSize)
	utils.ThrowErr(err)

	result := utils.NewStats(batchSize)
	resultExp := utils.NewStats(batchSize)

	iters := 0
	maxIters := 5

	for true {
		X, Y, err := datacn.Batch()
		Xbatch := plainUtils.NewDense(X)
		if err != nil || iters >= maxIters {
			//dataset completed
			break
		}
		Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
		utils.ThrowErr(err)
		cipherUtils.PrepackBlocks(Xenc, splitInfo.ColsOfWeights[0], Box)

		if !debug {
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
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
}

//EXPERIMENT 2 - Model encrypted,data in clear:
//In this scenario the model is sent by server to the client in encrypted form
//Server offers an oblivious decryption service
//EDIT: this version uses localhost to simulate LAN environment
func TestCryptonet_EvalBatchClearModelEnc(t *testing.T) {
	//13s for 41 batch with logn14 mask

	var debug = true       //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	loader := new(CNLoader)
	cn := loader.Load("cryptonet_packed.json")

	params := paramsLogN14Mask
	features := 28 * 28
	rows, cols := cn.GetDimentions()
	possibleSplits := cipherUtils.FindSplits(-1, features, rows, cols, params)

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}

	serverAddr := "127.0.0.1:8001"
	client, err := distributed.NewClient(serverAddr, Box, poolSize, true)
	utils.ThrowErr(err)
	server, err := distributed.NewServer(Box, serverAddr, true)
	go server.Listen()
	utils.ThrowErr(err)

	splits := possibleSplits[0]
	cipherUtils.PrintSetOfSplits(splits)

	splitInfo, _ := cipherUtils.ExctractInfo(splits)

	batchSize := splitInfo.InputRows * splitInfo.InputRowP
	cn.SetBatch(batchSize)

	Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

	cne := cn.NewHE(splits, true, false, 4, params.MaxLevel(), nil, poolSize, Box)

	fmt.Println("Encrypted Cryptonet...")

	datacn := data.LoadData("cryptonet_data_nopad.json")
	err = datacn.Init(batchSize)
	utils.ThrowErr(err)

	result := utils.NewStats(batchSize)
	resultExp := utils.NewStats(batchSize)

	iters := 0
	maxIters := 5

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
			resHE, end := cne.Eval(Xp)
			resMasked := client.StartProto(distributed.MASKING, resHE)
			fmt.Println("End ", end)

			resClear := cipherUtils.DecodeInput(resMasked, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		} else {
			resHE, resExp, end := cne.EvalDebug(Xp, Xbatch, cn, 1.0)
			resMasked := client.StartProto(distributed.MASKING, resHE)
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
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
}

//EXPERIMENT 2 - Model encrypted,data in clear:
//In this scenario the model is sent by server to the client in encrypted form
//Server offers an oblivious decryption service
//EDIT: this version spawns the server on a server on the iccluster
func TestCryptonet_EvalBatchClearModelEnc_LAN(t *testing.T) {
	//13s for 41 batch with logn14 mask

	var debug = true       //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	loader := new(CNLoader)
	cn := loader.Load("cryptonet_packed.json")

	params := paramsLogN14Mask
	features := 28 * 28
	rows, cols := cn.GetDimentions()
	possibleSplits := cipherUtils.FindSplits(-1, features, rows, cols, params)

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}

	serverAddr := cluster.SubNet + strconv.Itoa(cluster.StartingAddr+1) + ":" + strconv.Itoa(distributed.ServicePort)
	setupServerAddr := cluster.SubNet + strconv.Itoa(cluster.StartingAddr+1) + ":" + strconv.Itoa(distributed.SetupPort)
	client, err := distributed.NewClient(serverAddr, Box, poolSize, false)
	utils.ThrowErr(err)
	client.ClientSetup(setupServerAddr, params, Box.Sk)
	utils.ThrowErr(err)

	splits := possibleSplits[0]
	cipherUtils.PrintSetOfSplits(splits)

	splitInfo, _ := cipherUtils.ExctractInfo(splits)

	batchSize := splitInfo.InputRows * splitInfo.InputRowP
	cn.SetBatch(batchSize)

	Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

	cne := cn.NewHE(splits, true, false, 4, params.MaxLevel(), nil, poolSize, Box)

	fmt.Println("Encrypted Cryptonet...")

	datacn := data.LoadData("cryptonet_data_nopad.json")
	err = datacn.Init(batchSize)
	utils.ThrowErr(err)

	result := utils.NewStats(batchSize)
	resultExp := utils.NewStats(batchSize)

	iters := 0
	maxIters := 5

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
			resHE, end := cne.Eval(Xp)
			resMasked := client.StartProto(distributed.MASKING, resHE)
			fmt.Println("End ", end)

			resClear := cipherUtils.DecodeInput(resMasked, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
		} else {
			resHE, resExp, end := cne.EvalDebug(Xp, Xbatch, cn, 1.0)
			resMasked := client.StartProto(distributed.MASKING, resHE)
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
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
}
