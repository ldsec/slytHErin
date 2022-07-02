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
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"runtime"
	"strconv"
	"testing"
)

var paramsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogSlots:     13,
	LogQ:         []int{40, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
	LogP:         []int{55},
	DefaultScale: 1 << 31,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

//Given a deg of approximation of 63 (so 6 level needed for evaluation) this set of params performs really good:
//It has 18 levels, so it invokes a bootstrap every 2 layers (1 lvl for mul + 6 lvl for activation) when the level
//is 4, which is the minimum level. In this case, bootstrap is called only when needed
//In case of NN50, cut the modulo chain at 11 levels, so to spare memory. In this case Btp happens every layer
var paramsLogN15_NN20, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})
var paramsLogN15_NN50, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

func TestEvalDataEncModelEnc(t *testing.T) {
	debug := false
	multithread := true
	poolSize := 1
	if multithread {
		poolSize = runtime.NumCPU()
	}
	fmt.Printf("Running on %d threads\n", poolSize)
	layers := 50 //20 or 50

	nn := LoadNN("nn" + strconv.Itoa(layers) + "_packed.json")

	nn.Init(layers, false)

	// CRYPTO
	ckksParams := bootstrapping.N15QP768H192H32.SchemeParams
	btpParams := bootstrapping.N15QP768H192H32.BootstrappingParams
	btpCapacity := 2 //param dependent
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	utils.ThrowErr(err)

	weightRows := make([]int, layers+1)
	weightCols := make([]int, layers+1)
	weightRows[0] = 784
	weightCols[0] = 676
	weightRows[1] = 676
	weightCols[1] = 92
	for i := 2; i < layers+1; i++ {
		weightRows[i] = 92
		weightCols[i] = 92
	}
	weightCols[layers] = 10
	possibleSplits := cipherUtils.FindSplits(-1, 784, weightRows, weightCols, params, 0.2, true, true)

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}
	for _, splits := range possibleSplits {
		cipherUtils.PrintSetOfSplits(splits)
		splitInfo := cipherUtils.ExctractInfo(splits)
		batchSize := splitInfo.InputRows * splitInfo.InputRowP
		fmt.Println("Batch: ", batchSize)
		Box := cipherUtils.NewBox(params)
		Box = cipherUtils.BoxWithEvaluators(Box, btpParams, true, splitInfo.InputRows, splitInfo.InputCols, splitInfo.NumWeights, splitInfo.RowsOfWeights, splitInfo.ColsOfWeights)
		Btp := cipherUtils.NewBootstrapper(Box, poolSize)

		weights, biases := nn.BuildParams(batchSize)
		weightsRescaled, biasesRescaled := nn.RescaleWeightsForActivation(weights, biases)
		nne, err := nn.EncryptNN(weightsRescaled, biasesRescaled, splits, btpCapacity, -1, Box, poolSize)
		utils.ThrowErr(err)
		//load dataset
		dataSn := data.LoadData("nn_data.json")
		err = dataSn.Init(batchSize)
		if err != nil {
			fmt.Println(err)
			return
		}
		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := 2

		var elapsed int64
		var res utils.Stats
		for true {
			X, Y, err := dataSn.Batch()
			Xbatch := plainUtils.NewDense(X)
			if err != nil || iters >= maxIters {
				//dataset completed
				break
			}
			Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), Box)
			utils.ThrowErr(err)
			//res := sn.EvalBatchEncryptedCompressed(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10, false)
			if !debug {
				res = nne.EvalBatchEncrypted(Xenc, Y, 10, Btp)
			} else {
				res = nne.EvalBatchEncrypted_Debug(Xenc, Y, Xbatch, weights, biases, utils.SoftReLu, 10, Btp)
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
func TestEvalDataEncModelEnc_Distributed(t *testing.T) {
	//NN20
	//5 parties -> 146 batch in 229s
	//10 parties ->
	debug := false
	multithread := true
	poolSize := 1
	if multithread {
		poolSize = runtime.NumCPU()
	}
	fmt.Printf("Running on %d threads\n", poolSize)
	layers := 20 //20 or 50

	nn := LoadNN("nn" + strconv.Itoa(layers) + "_packed.json")

	nn.Init(layers, true)

	var params ckks.Parameters

	params = paramsLogN14

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

	//Create distributed parties
	localhost := "127.0.0.1"
	partiesAddr := make([]string, parties)
	for i := 0; i < parties; i++ {
		if i == 0 {
			partiesAddr[i] = localhost + ":" + strconv.Itoa(8000)
		} else {
			partiesAddr[i] = localhost + ":" + strconv.Itoa(8080+i)
		}
	}

	weightRows := make([]int, layers+1)
	weightCols := make([]int, layers+1)
	weightRows[0] = 784
	weightCols[0] = 676
	weightRows[1] = 676
	weightCols[1] = 92
	for i := 2; i < layers+1; i++ {
		weightRows[i] = 92
		weightCols[i] = 92
	}
	weightCols[layers] = 10
	possibleSplits := cipherUtils.FindSplits(292, 784, weightRows, weightCols, params, 0.2, true, true)

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}
	for _, splits := range possibleSplits {
		splitInfo := cipherUtils.ExctractInfo(splits)
		cipherUtils.PrintSetOfSplits(splits)
		batchSize := splitInfo.InputRows * splitInfo.InputRowP
		rotations := cipherUtils.GenRotations(splitInfo.InputRows, splitInfo.InputCols, splitInfo.NumWeights, splitInfo.RowsOfWeights, splitInfo.ColsOfWeights, params, nil)
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

		//start distributed parties
		master, err := distributed.NewLocalMaster(skShares[0], pkP, params, parties, partiesAddr, Box, poolSize)
		utils.ThrowErr(err)
		players := make([]*distributed.LocalPlayer, parties-1)
		//start players
		for i := 0; i < parties-1; i++ {
			players[i], err = distributed.NewLocalPlayer(skShares[i+1], pkP, params, i+1, partiesAddr[i+1])
			go players[i].Listen()
			utils.ThrowErr(err)
		}

		// info for bootstrapping
		var minLevel int
		var ok bool
		if minLevel, _, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q()); ok != true || minLevel > params.MaxLevel() {
			utils.ThrowErr(errors.New("Not enough levels to ensure correcness and 128 security"))
		}

		weights, biases := nn.BuildParams(batchSize)
		weightsRescaled, biasesRescaled := nn.RescaleWeightsForActivation(weights, biases)
		nne, err := nn.EncryptNN(weightsRescaled, biasesRescaled, splits, -1, minLevel, Box, poolSize)
		utils.ThrowErr(err)
		//load dataset
		dataSn := data.LoadData("nn_data.json")
		err = dataSn.Init(batchSize)
		if err != nil {
			fmt.Println(err)
			return
		}
		corrects := 0
		accuracy := 0.0

		tot := 0
		iters := 0
		maxIters := 2

		var elapsed int64
		var res utils.Stats
		for true {
			X, Y, err := dataSn.Batch()
			Xbatch := plainUtils.NewDense(X)
			if err != nil || iters >= maxIters {
				//dataset completed
				break
			}
			Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), Box)
			utils.ThrowErr(err)
			//res := sn.EvalBatchEncryptedCompressed(Xbatch, Y, Xenc, weightsBlock, biasBlock, Box, 10, false)
			if !debug {
				res = nne.EvalBatchEncrypted_Distributed(Xenc, Y, 10, pkQ, decQ, minLevel, master)
			} else {
				res = nne.EvalBatchEncrypted_Distributed_Debug(Xenc, Y, Xbatch, weights, biases, utils.SoftReLu, 10, pkQ, decQ, minLevel, master)
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
