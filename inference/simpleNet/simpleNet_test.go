package simpleNet

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/data"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"math"
	"runtime"
	"testing"
)
import "github.com/tuneinsight/lattigo/v3/ckks"
import "github.com/tuneinsight/lattigo/v3/rlwe"

var paramsLogN15, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogQ:         []int{45, 35, 35, 35, 35, 35, 35}, //Log(PQ)
	LogP:         []int{50, 50, 50, 50},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     14,
	DefaultScale: float64(1 << 35),
})

var paramsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{35, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{42, 42},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 30),
})

var paramsLogN13, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         13,
	LogQ:         []int{29, 26, 26, 26, 26, 26, 26}, //Log(PQ) <= 218 for LogN 13
	LogP:         []int{33},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     12,
	DefaultScale: float64(1 << 26),
})

func TestSimpleNetEcd_EvalBatchEncrypted(t *testing.T) {

	var debug = false      //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	sn := LoadSimpleNet("simplenet_packed.json")
	sn.Init()

	params := paramsLogN13

	Box := cipherUtils.NewBox(params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
	}

	possibleSplits := cipherUtils.FindSplits(-1, 28*28, []int{784, 100}, []int{100, 10}, params, 0.5, false, false)

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

		Box = cipherUtils.BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, splitInfo.InputRows, splitInfo.InputCols, splitInfo.NumWeights, splitInfo.RowsOfWeights, splitInfo.ColsOfWeights)

		sne := sn.EncodeSimpleNet(weights, biases, splits, Box, poolSize)

		fmt.Println("Encoded SimpleNet...")

		dataSn := data.LoadData("simplenet_data_nopad.json")
		err := dataSn.Init(batchSize)
		utils.ThrowErr(err)

		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := int(math.Ceil(float64(1024 / batchSize)))
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
			cipherUtils.PrepackBlocks(Xenc, sne.Weights[0].InnerCols, Box.Evaluator)
			//res := sn.EvalBatchEncryptedCompressed(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10, false)
			if !debug {
				res = sne.EvalBatchEncrypted(Xenc, Y, 10)
			} else {
				weightsD, biasesD := sn.CompressLayers(weights, biases)
				res = sne.EvalBatchEncrypted_Debug(Xenc, Xbatch, weightsD, biasesD, sn.ReLUApprox, Y, 10)
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
