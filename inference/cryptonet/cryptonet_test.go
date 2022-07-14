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
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{42, 42},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

//Model in clear - data encrypted
func TestCryptonetEcd_EvalBatchEncrypted(t *testing.T) {

	//41 in 6.14s with logn14
	//83 in 10.5s with logn15

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	sn := Loadcryptonet("cryptonet_packed.json")
	sn.Init()

	params := paramsLogN14
	possibleSplits := cipherUtils.FindSplits(40, 28*28, []int{784, 720, 100}, []int{720, 100, 10}, params)

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

		splitInfo := cipherUtils.ExctractInfo(splits)

		batchSize := splitInfo.InputRows * splitInfo.InputRowP

		weights, biases := sn.BuildParams(batchSize)

		Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

		sne := sn.Encodecryptonet(weights, biases, splits, Box, poolSize)

		fmt.Println("Encoded cryptonet...")

		dataSn := data.LoadData("cryptonet_data_nopad.json")
		err := dataSn.Init(batchSize)
		utils.ThrowErr(err)

		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := 5
		var elapsed int64
		var res utils.Stats
		for true {
			X, Y, err := dataSn.Batch()
			Xbatch := plainUtils.NewDense(X)
			if err != nil || iters >= maxIters {
				//dataset completed
				break
			}
			Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
			utils.ThrowErr(err)
			cipherUtils.PrepackBlocks(Xenc, sne.Weights[0].InnerCols, Box)

			if !debug {
				res = sne.EvalBatchEncrypted(Xenc, Y, 10)
			} else {
				res = sne.EvalBatchEncrypted_Debug(Xenc, Xbatch, weights, biases, sn.ReLUApprox, Y, 10)
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

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	sn := Loadcryptonet("cryptonet_packed.json")
	sn.Init()

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

		splitInfo := cipherUtils.ExctractInfo(splits)

		batchSize := splitInfo.InputRows * splitInfo.InputRowP

		weights, biases := sn.BuildParams(batchSize)

		Box = cipherUtils.BoxWithSplits(Box, bootstrapping.Parameters{}, false, splits)

		sne := sn.Encryptcryptonet(weights, biases, splits, Box, poolSize)

		fmt.Println("Encrypted cryptonet...")

		dataSn := data.LoadData("cryptonet_data_nopad.json")
		err := dataSn.Init(batchSize)
		utils.ThrowErr(err)

		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := 5
		var elapsed int64
		var res utils.Stats
		for true {
			X, Y, err := dataSn.Batch()
			Xbatch := plainUtils.NewDense(X)
			if err != nil || iters >= maxIters {
				//dataset completed
				break
			}
			Xp, err := cipherUtils.NewPlainInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
			utils.ThrowErr(err)
			cipherUtils.PrepackBlocks(Xp, sne.Weights[0].InnerCols, Box)

			if !debug {
				res = sne.EvalBatchWithModelEnc(Xp, Y, 10)
			} else {
				res = sne.EvalBatchWithModelEnc_Debug(Xp, Xbatch, weights, biases, sn.ReLUApprox, Y, 10)
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
