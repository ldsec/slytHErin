package cryptonet

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"runtime"
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
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{42, 42},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

//Model in clear - data encrypted
func TestCryptonetEcd_EvalBatchEncrypted(t *testing.T) {

	//logN 14 -> 7.6s for 41
	//83 in 10.5s with logn15

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	cn := LoadCryptonet("cryptonet_packed.json")
	cn.Init()

	params := paramsLogN14
	possibleSplits := cipherUtils.FindSplits(-1, 28*28, []int{784, 720, 100}, []int{720, 100, 10}, params)

	//params := paramsLogN15
	//possibleSplits := cipherUtils.FindSplits(-1, 28*28, []int{784, 720, 100}, []int{720, 100, 10}, params, 0.0, true, true)

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}
	cipherUtils.PrintAllSplits(possibleSplits)

	for _, splits := range possibleSplits {
		fmt.Println()
		fmt.Println("Trying split: ")
		fmt.Println()
		cipherUtils.PrintSetOfSplits(splits)

		splitInfo, _ := cipherUtils.ExctractInfo(splits)

		batchSize := splitInfo.InputRows * splitInfo.InputRowP

		weights, biases := cn.BuildParams(batchSize)

		Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

		cne := cn.EncodeCryptonet(weights, biases, splits, Box, poolSize)

		fmt.Println("Encoded Cryptonet...")

		datacn := data.LoadData("cryptonet_data_nopad.json")
		err := datacn.Init(batchSize)
		utils.ThrowErr(err)

		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := 5
		var elapsed int64
		var res utils.Stats
		for true {
			X, Y, err := datacn.Batch()
			Xbatch := plainUtils.NewDense(X)
			if err != nil || iters >= maxIters {
				//dataset completed
				break
			}
			Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
			utils.ThrowErr(err)
			cipherUtils.PrepackBlocks(Xenc, cne.Weights[0].InnerCols, Box)

			if !debug {
				res = cne.EvalBatchEncrypted(Xenc, Y, 10)
			} else {
				wR, bR := cn.RescaleForActivation(weights, biases)
				res = cne.EvalBatchEncrypted_Debug(Xenc, Xbatch, wR, bR, cn.ReLUApprox, Y, 10)
			}
			fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
			fmt.Println("Accuracy:", res.Accuracy)
			corrects += res.Corrects
			accuracy += res.Accuracy
			tot += batchSize
			elapsed += res.Time.Milliseconds()
			iters++
			fmt.Println()
		}
		fmt.Println("Accuracy:", accuracy/float64(iters))
		fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
	}
}

//Model encrypted - data in clear
func TestCryptonetEnc_EvalBatchWithModelEnc(t *testing.T) {

	//logN 14 -> 9.36 for 41
	var debug = true       //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	cn := LoadCryptonet("cryptonet_packed.json")
	cn.Init()

	params := paramsLogN14
	possibleSplits := cipherUtils.FindSplits(-1, 28*28, []int{784, 720, 100}, []int{720, 100, 10}, params)

	//params := paramsLogN15
	//possibleSplits := cipherUtils.FindSplits(-1, 28*28, []int{784, 720, 100}, []int{720, 100, 10}, params, 0.0, true, true)

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}
	cipherUtils.PrintAllSplits(possibleSplits)

	for _, splits := range possibleSplits {
		fmt.Println()
		fmt.Println("Trying split: ")
		fmt.Println()
		cipherUtils.PrintSetOfSplits(splits)

		splitInfo, _ := cipherUtils.ExctractInfo(splits)

		batchSize := splitInfo.InputRows * splitInfo.InputRowP

		weights, biases := cn.BuildParams(batchSize)

		Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

		cne := cn.EncryptCryptonet(weights, biases, splits, Box, poolSize)

		fmt.Println("Encrypted Cryptonet...")

		datacn := data.LoadData("cryptonet_data_nopad.json")
		err := datacn.Init(batchSize)
		utils.ThrowErr(err)

		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := 5
		var elapsed int64
		var res utils.Stats
		for true {
			X, Y, err := datacn.Batch()
			Xbatch := plainUtils.NewDense(X)
			if err != nil || iters >= maxIters {
				//dataset completed
				break
			}
			Xp, err := cipherUtils.NewPlainInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
			utils.ThrowErr(err)
			cipherUtils.PrepackBlocks(Xp, cne.Weights[0].InnerCols, Box)

			if !debug {
				res = cne.EvalBatchWithModelEnc(Xp, Y, 10)
			} else {
				wR, bR := cn.RescaleForActivation(weights, biases)
				res = cne.EvalBatchWithModelEnc_Debug(Xp, Xbatch, wR, bR, cn.ReLUApprox, Y, 10)
			}
			fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
			fmt.Println("Accuracy:", res.Accuracy)
			corrects += res.Corrects
			accuracy += res.Accuracy
			tot += batchSize
			elapsed += res.Time.Milliseconds()
			iters++
			fmt.Println()
		}
		fmt.Println("Accuracy:", accuracy/float64(iters))
		fmt.Println("Latency(avg ms per batch):", float64(elapsed)/float64(iters))
	}
}
