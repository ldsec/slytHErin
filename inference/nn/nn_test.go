package nn

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/cluster"
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
	"os"
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

//Given a deg of approximation of 63 (so 6 level needed for evaluation) this set of params performs really good:
//It has 18 levels, so it invokes a bootstrap every 2 layers (1 lvl for mul + 6 lvl for activation) when the level
//is 4, which is the minimum level. In this case, bootstrap is called only when needed
//In case of NN50, cut the modulo chain at 11 levels, so to spare memory. In this case Btp happens every layer
var paramsLogN15_NN50, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         15,
	LogSlots:     14,
	LogQ:         []int{44, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
	LogP:         []int{50, 50, 50, 50},
	DefaultScale: 1 << 35,
	Sigma:        rlwe.DefaultSigma,
	RingType:     ring.Standard,
})

var paramsLogN16, _ = ckks.NewParametersFromLiteral(bootstrapping.N16QP1546H192H32.SchemeParams)
var btpParamsLogN16 = bootstrapping.N16QP1546H192H32.BootstrappingParams

//EXPERIMENT 1 - Model clear,data encrypted, Centralized Bootstrapping
//Querier sends encrypted data to server for privacy-preserving inference. Server uses centralized bootstrapping
//Uses NN50
func TestNN_EvalBatchEncrypted_CentralizedBtp(t *testing.T) {
	//nn50 - 38m for 96 batch
	var HETrain = false    //model trained with HE SGD, LSE and poly act (HE Friendly)
	var layers = 50        //20 or 50
	var debug = true       //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	suffix := "_poly"
	if !HETrain {
		suffix = ""
	}
	path := fmt.Sprintf("nn%d%s_packed.json", layers, suffix)

	loader := new(NNLoader)
	nn := loader.Load(path)

	params := paramsLogN16
	btpParams := btpParamsLogN16

	features := 28 * 28
	rows, cols := nn.GetDimentions()
	possibleSplits := cipherUtils.FindSplits(-1, features, rows, cols, params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}
	cipherUtils.PrintAllSplits(possibleSplits)

	for _, splits := range possibleSplits {
		cipherUtils.PrintSetOfSplits(splits)

		splitInfo, splitCode := cipherUtils.ExctractInfo(splits)

		batchSize := splitInfo.InputRows * splitInfo.InputRowP
		nn.SetBatch(batchSize)

		Box := cipherUtils.NewBox(params)

		path := fmt.Sprintf("/root/nn%d_centralized_logN%dlogPQ%d__%s", layers, params.LogN(), params.LogP()+params.LogQ(), splitCode)
		fmt.Println("Key path: ", path)
		if _, err := os.Stat(path + "_sk"); errors.Is(err, os.ErrNotExist) {
			fmt.Println("Creating rotation keys...")
			Box = cipherUtils.BoxWithSplits(Box, btpParams, true, splits)
			fmt.Println("Created rotation keys...")
			cipherUtils.SerializeBox(path, Box)
		} else {
			fmt.Println("Reading keys from disk")
			Box = cipherUtils.DeserealizeBox(path, params, btpParams, true)
		}

		Btp := cipherUtils.NewBootstrapper(Box, poolSize)
		cne := nn.NewHE(splits, false, true, 0, 9, Btp, poolSize, Box)

		fmt.Println("Encoded NN...")

		datacn := data.LoadData("nn_data.json")
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
				fmt.Println("Accuracy HE: ", accuracy)
				result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})

			} else {
				resHE, resExp, end := cne.EvalDebug(Xenc, Xbatch, nn, 1.5)
				fmt.Println("End", end)
				resClear := cipherUtils.DecInput(resHE, Box)
				corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
				fmt.Println("Accuracy HE: ", accuracy)
				result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})
				corrects, accuracy, _ = utils.Predict(Y, 10, plainUtils.MatToArray(resExp))
				fmt.Println("Accuracy Expected: ", accuracy)
			}
			iters++
		}
		result.PrintResult()
		if debug {
			fmt.Println("Expected")
			resultExp.PrintResult()
		}
		break
	}
}

//EXPERIMENT 2: Model encrypted,data encrypted,distributed bootstrap
//The model is supposed to be trained under encryption.
//Querier sends data encrypted under nodes cohort public key to master server which performs computation
//Master invokes distributed refresh with a variable number of parties, and finally invokes key switch protocol
//Prediction is received under Querier public key
//Use NN20_poly since it was trained with HE friendly parameters
//EDIT: this version emulates LAN on localhost with injected sleep calls for latency and bandwidth
func TestNN20_EvalBatchEncrypted_DistributedBtp(t *testing.T) {
	//nn20 - 4m for 48 batch with logN15_NN20

	var HETrain = true //model trained with HE SGD, LSE and poly act (HE Friendly)
	var layers = 20    //20 or 50
	var parties = 10
	var debug = true       //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	suffix := "_poly"
	if !HETrain {
		suffix = ""
	}
	path := fmt.Sprintf("nn%d%s_packed.json", layers, suffix)

	loader := new(NNLoader)
	nn := loader.Load(path)

	params := paramsLogN15_NN20

	features := 28 * 28
	rows, cols := nn.GetDimentions()
	possibleSplits := cipherUtils.FindSplits(-1, features, rows, cols, params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}
	cipherUtils.PrintAllSplits(possibleSplits)

	// QUERIER
	kgenQ := ckks.NewKeyGenerator(params)
	skQ := kgenQ.GenSecretKey()
	pkQ := kgenQ.GenPublicKey(skQ)
	decQ := ckks.NewDecryptor(params, skQ)

	// PARTIES
	// [!] All the keys for encryption, keySw, Relin can be produced by MPC protocols
	// [!] We assume that these protocols have been run in a setup phase by the parties

	//Allocate addresses
	localhost := "127.0.0.1"
	partiesAddr := make([]string, parties)
	for i := 0; i < parties; i++ {
		if i == 0 {
			partiesAddr[i] = localhost + ":" + strconv.Itoa(8000)
		} else {
			partiesAddr[i] = localhost + ":" + strconv.Itoa(8080+i)
		}
	}

	splits := possibleSplits[0]
	splitInfo, splitCode := cipherUtils.ExctractInfo(splits)
	cipherUtils.PrintSetOfSplits(splits)
	batchSize := splitInfo.InputRows * splitInfo.InputRowP
	nn.SetBatch(batchSize)

	path = fmt.Sprintf("/root/nn%d_parties%d_logN%dlogPQ%d__%s", layers, parties, params.LogN(), params.LogP()+params.LogQ(), splitCode)
	crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})

	skP := new(rlwe.SecretKey)
	skShares := make([]*rlwe.SecretKey, parties)
	pkP := new(rlwe.PublicKey)
	rtks := new(rlwe.RotationKeySet)
	rlk := new(rlwe.RelinearizationKey)
	kgenP := ckks.NewKeyGenerator(params)

	if _, err := os.Stat(path + "_sk"); errors.Is(err, os.ErrNotExist) {
		skShares, skP, pkP, kgenP = distributed.DummyEncKeyGen(params, crs, parties)
		rlk = distributed.DummyRelinKeyGen(params, crs, skShares)

		rotations := cipherUtils.GenRotations(splitInfo.InputRows, splitInfo.InputCols, splitInfo.NumWeights, splitInfo.RowsOfWeights, splitInfo.ColsOfWeights, splitInfo.RowPOfWeights, splitInfo.ColPOfWeights, params, nil)
		rtks = kgenP.GenRotationKeysForRotations(rotations, true, skP)

		distributed.SerializeKeys(skP, skShares, rtks, path) //write to file
	} else {
		skP, skShares, rtks = distributed.DeserializeKeys(path, parties) //read from file
		kgenP = ckks.NewKeyGenerator(params)
		pkP = kgenP.GenPublicKey(skP)
		rlk = distributed.DummyRelinKeyGen(params, crs, skShares)
	}

	decP := ckks.NewDecryptor(params, skP)

	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),                                             //public
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}), //from parties
		Decryptor:    decP,                                                                //from parties for debug
		Encryptor:    ckks.NewEncryptor(params, pkP),                                      //from parties
		BootStrapper: nil,
	}

	//[!] Start distributed parties
	master, err := distributed.NewLocalMaster(skShares[0], pkP, params, parties, partiesAddr, Box, poolSize, true)
	utils.ThrowErr(err)
	players := make([]*distributed.LocalPlayer, parties-1)
	//start players
	for i := 0; i < parties-1; i++ {
		players[i], err = distributed.NewLocalPlayer(skShares[i+1], pkP, params, i+1, partiesAddr[i+1], true)
		go players[i].Listen()
		utils.ThrowErr(err)
	}

	// info for bootstrapping
	var minLevel int
	var ok bool
	if minLevel, _, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q()); ok != true || minLevel > params.MaxLevel() {
		utils.ThrowErr(errors.New("Not enough levels to ensure correcness and 128 security"))
	}

	Btp := distributed.NewDistributedBootstrapper(master, poolSize)
	cne := nn.NewHE(splits, true, true, minLevel, params.MaxLevel(), Btp, poolSize, Box)

	fmt.Println("Encryped NN...")

	datacn := data.LoadData("nn_data.json")
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
		Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
		utils.ThrowErr(err)
		cipherUtils.PrepackBlocks(Xenc, splitInfo.ColsOfWeights[0], Box)

		if !debug {
			resHE, end := cne.Eval(Xenc)

			fmt.Println("Key Switch to querier public key")
			master.StartProto(distributed.CKSWITCH, resHE, pkQ, minLevel)

			fmt.Println("End", end)
			//Switch to Dec of Querier
			Box.Decryptor = decQ
			resClear := cipherUtils.DecInput(resHE, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})

		} else {
			resHE, resExp, end := cne.EvalDebug(Xenc, Xbatch, nn, 1.0)

			fmt.Println("Key Switch to querier public key")
			master.StartProto(distributed.CKSWITCH, resHE, pkQ, minLevel)

			fmt.Println("End", end)
			//Switch to Dec of Querier
			Box.Decryptor = decQ
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

//EXPERIMENT 2: Model encrypted,data encrypted,distributed bootstrap
//The model is supposed to be trained under encryption.
//Querier sends data encrypted under nodes cohort public key to master server which performs computation
//Master invokes distributed refresh with a variable number of parties, and finally invokes key switch protocol
//Prediction is received under Querier public key
//Use NN20_poly since it was trained with HE friendly parameters
//EDIT: this version attempts to distribute the workload among real servers in LAN setting
func TestNN20_EvalBatchEncrypted_DistributedBtp_LAN(t *testing.T) {
	//291s for 48 batch, loss in accuracy ~ 0.4%

	var HETrain = true //model trained with HE SGD, LSE and poly act (HE Friendly)
	var layers = 20
	var debug = true       //set to true for debug mode
	var multiThread = true //set to true to enable multiple threads

	suffix := "_poly"
	if !HETrain {
		suffix = ""
	}
	path := fmt.Sprintf("nn%d%s_packed.json", layers, suffix)

	loader := new(NNLoader)
	nn := loader.Load(path)

	params := paramsLogN15_NN20

	features := 28 * 28
	rows, cols := nn.GetDimentions()
	possibleSplits := cipherUtils.FindSplits(-1, features, rows, cols, params)

	poolSize := 1
	if multiThread {
		poolSize = runtime.NumCPU()
		fmt.Println("Num VCPUs: ", poolSize)
	}

	if len(possibleSplits) == 0 {
		panic(errors.New("No splits found!"))
	}

	// QUERIER
	kgenQ := ckks.NewKeyGenerator(params)
	skQ := kgenQ.GenSecretKey()
	pkQ := kgenQ.GenPublicKey(skQ)
	decQ := ckks.NewDecryptor(params, skQ)

	// PARTIES
	// [!] All the keys for encryption, keySw, Relin can be produced by MPC protocols
	// [!] We assume that these protocols have been run in a setup phase by the parties

	//Allocate addresses on ICC LAN
	clusterConfig := cluster.ReadConfig("../cluster/config.json")
	parties := clusterConfig.NumServers
	partiesAddr := make([]string, parties)
	for i := 0; i < parties; i++ {
		partiesAddr[i] = clusterConfig.ClusterIps[i]
	}

	splits := possibleSplits[0]
	splitInfo, splitCode := cipherUtils.ExctractInfo(splits)
	cipherUtils.PrintSetOfSplits(splits)
	batchSize := splitInfo.InputRows * splitInfo.InputRowP
	nn.SetBatch(batchSize)

	path = fmt.Sprintf("/root/nn%d_parties%d_logN%dlogPQ%d__%s", layers, parties, params.LogN(), params.LogP()+params.LogQ(), splitCode)
	crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})

	skP := new(rlwe.SecretKey)
	skShares := make([]*rlwe.SecretKey, parties)
	pkP := new(rlwe.PublicKey)
	rtks := new(rlwe.RotationKeySet)
	rlk := new(rlwe.RelinearizationKey)
	kgenP := ckks.NewKeyGenerator(params)

	if _, err := os.Stat(path + "_sk"); errors.Is(err, os.ErrNotExist) {
		skShares, skP, pkP, kgenP = distributed.DummyEncKeyGen(params, crs, parties)
		rlk = distributed.DummyRelinKeyGen(params, crs, skShares)

		rotations := cipherUtils.GenRotations(splitInfo.InputRows, splitInfo.InputCols, splitInfo.NumWeights, splitInfo.RowsOfWeights, splitInfo.ColsOfWeights, splitInfo.RowPOfWeights, splitInfo.ColPOfWeights, params, nil)
		rtks = kgenP.GenRotationKeysForRotations(rotations, true, skP)

		distributed.SerializeKeys(skP, skShares, rtks, path) //write to file
	} else {
		skP, skShares, rtks = distributed.DeserializeKeys(path, parties) //read from file
		kgenP = ckks.NewKeyGenerator(params)
		pkP = kgenP.GenPublicKey(skP)
		rlk = distributed.DummyRelinKeyGen(params, crs, skShares)
	}

	decP := ckks.NewDecryptor(params, skP)

	Box := cipherUtils.CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),                                             //public
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks}), //from parties
		Decryptor:    decP,                                                                //from parties for debug
		Encryptor:    ckks.NewEncryptor(params, pkP),                                      //from parties
		BootStrapper: nil,
	}

	//[!] Start distributed parties
	master, err := distributed.NewLocalMaster(skShares[0], pkP, params, parties, partiesAddr, Box, poolSize, false)
	utils.ThrowErr(err)
	master.MasterSetup(partiesAddr, parties, skShares, pkP)

	// info for bootstrapping
	var minLevel int
	var ok bool
	if minLevel, _, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q()); ok != true || minLevel > params.MaxLevel() {
		utils.ThrowErr(errors.New("Not enough levels to ensure correcness and 128 security"))
	}

	Btp := distributed.NewDistributedBootstrapper(master, poolSize)
	//instantiate new he network
	cne := nn.NewHE(splits, true, true, minLevel, params.MaxLevel(), Btp, poolSize, Box)

	fmt.Println("Encryped NN...")

	datacn := data.LoadData("nn_data.json")
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
		Xenc, err := cipherUtils.NewEncInput(Xbatch, splitInfo.InputRowP, splitInfo.InputColP, params.MaxLevel(), params.DefaultScale(), Box)
		utils.ThrowErr(err)
		cipherUtils.PrepackBlocks(Xenc, splitInfo.ColsOfWeights[0], Box)

		if !debug {
			resHE, end := cne.Eval(Xenc)

			fmt.Println("Key Switch to querier public key")
			master.StartProto(distributed.CKSWITCH, resHE, pkQ, minLevel)

			fmt.Println("End", end)
			//Switch to Dec of Querier
			Box.Decryptor = decQ
			resClear := cipherUtils.DecInput(resHE, Box)
			corrects, accuracy, _ := utils.Predict(Y, 10, resClear)
			fmt.Println("Accuracy HE: ", accuracy)
			result.Accumulate(utils.Stats{Corrects: corrects, Accuracy: accuracy, Time: end.Milliseconds()})

		} else {
			resHE, resExp, end := cne.EvalDebug(Xenc, Xbatch, nn, 2.0)

			fmt.Println("Key Switch to querier public key")
			master.StartProto(distributed.CKSWITCH, resHE, pkQ, minLevel)

			fmt.Println("End", end)
			//Switch to Dec of Querier
			Box.Decryptor = decQ
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
	master.StartProto(distributed.END, nil, nil, 0)
	result.PrintResult()
	fmt.Println()
	if debug {
		fmt.Println("Expected")
		resultExp.PrintResult()
	}
}
