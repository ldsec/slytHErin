package cryptonet

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/data"
	md "github.com/ldsec/dnn-inference/inference/multidim"
	pu "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	"math"
	"sync"
	"testing"
)

/*
//Testing cryptonet with MultiDimentional packing for enhanced throughput
func Test_BatchEncrypted(t *testing.T) {
	debug := false
	multithread := true
	if !multithread {
		//SINGLE THREAD

		sn := Loadcryptonet("cryptonet_packed.json")
		sn.Init()
		//crypto
		//8 levels neeeded
		ckksParams := ckks2.ParametersLiteral{
			LogN:     14,
			LogQ:     []int{35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
			LogP:     []int{37, 37, 37},
			Sigma:    rlwe2.DefaultSigma,
			LogSlots: 13,
			Scale:    float64(1 << 30),
		}

		params, err := ckks2.NewParametersFromLiteral(ckksParams)

		features := 784 //MNIST
		batchSize := 256
		innerDim := int(math.Ceil(float64(params.N()) / (2.0 * float64(batchSize))))
		fmt.Printf("Input Dense: Rows %d, Cols %d --> InnerDim: %d\n", batchSize, features, innerDim)
		dataSn := data.LoadData("cryptonet_data_nopad.json")
		err = dataSn.Init(batchSize)

		X, Y, err := dataSn.Batch()
		Xpacked := PackBatchParallel(pu.NewDense(X), innerDim, params)

		weightMatrices, biasMatrices := sn.BuildParams(batchSize)
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
		maxIters := 5
		var elapsed int64

		for true {
			Xenc := batchEnc.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Xpacked)
			fmt.Printf("Input dimentions:\nRows:%d\nCols:%d\nInnerDim:%d\nBatches:%d\n\n", Xenc.Rows(), Xenc.Cols(), Xenc.Dim(), Xpacked.Batches())
			var res utils.Stats
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
	} else {
		//THREADING
		poolsize := 3
		fmt.Println("vCPUS: ", poolsize)
		features := 784 //MNIST
		batchSize := 512

		dataSn := data.LoadData("cryptonet_data_nopad.json")
		err := dataSn.Init(batchSize)
		utils.ThrowErr(err)
		poolElapsed := make([]int64, poolsize)
		poolAcc := make([]float64, poolsize)
		poolBatches := batchSize * poolsize

		var wg sync.WaitGroup
		for i := 0; i < poolsize; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()

				fmt.Printf("[!] Thread %d started\n", i+1)

				sn := Loadcryptonet("cryptonet_packed.json")
				sn.Init()

				//10 levels neeeded
				ckksParams := ckks2.ParametersLiteral{
					LogN:     14,
					LogQ:     []int{35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
					LogP:     []int{37, 37, 37},
					Sigma:    rlwe2.DefaultSigma,
					LogSlots: 13,
					Scale:    float64(1 << 30),
				}

				params, err := ckks2.NewParametersFromLiteral(ckksParams)
				innerDim := int(math.Ceil(float64(params.N()) / (2.0 * float64(batchSize))))
				fmt.Printf("Input Dense: Rows %d, Cols %d --> InnerDim: %d\n", batchSize, features, innerDim)

				X, Y, err := dataSn.Batch()
				if err != nil {
					fmt.Printf("[!] Thread %d -- Could not get a batch...exiting\n\n", i+1)
					return
				}
				Xpacked := PackBatchParallel(pu.NewDense(X), innerDim, params)

				weightMatrices, biasMatrices := sn.BuildParams(batchSize)
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

				var elapsed int64

				Xenc := batchEnc.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Xpacked)
				fmt.Printf("Input dimentions:\nRows:%d\nCols:%d\nInnerDim:%d\nBatches:%d\n\n", Xenc.Rows(), Xenc.Cols(), Xenc.Dim(), Xpacked.Batches())
				var res utils.Stats
				if !debug {
					res = snMD.EvalBatchEncrypted(Y, Xenc, Box)
				} else {
					res = snMD.EvalBatchEncrypted_Debug(Y, Xenc, Box, pu.NewDense(X), weightMatrices, biasMatrices)
				}
				fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
				corrects += res.Corrects
				elapsed = res.Time.Milliseconds()
				X, Y, err = dataSn.Batch()
				Xpacked = PackBatchParallel(pu.NewDense(X), innerDim, params)

				fmt.Printf("[!] Thread %d -- Accuracy %f\n:", i+1, float64(corrects)/float64(batchSize))
				fmt.Printf("[!] Thread %d -- Latency(ms) %f\n:", i+1, float64(elapsed))
				poolElapsed[i] = elapsed
				poolAcc[i] = float64(corrects) / float64(batchSize)
			}(i)
		}
		//Aggregate
		wg.Wait()

		var avgElapsed int64 = 0
		avgAcc := 0.0
		for i := range poolElapsed {
			avgElapsed += poolElapsed[i]
			avgAcc += poolAcc[i]
		}
		fmt.Println("TOT Samples: ", poolBatches)
		fmt.Println("TOT Latency (ms) :", float64(avgElapsed)/float64(poolsize))
		fmt.Println("TOT Accuracy :", float64(avgAcc)/float64(poolsize))
	}
}
*/
//Testing cryptonet with MultiDimentional packing for enhanced throughput
func Test_BatchEncrypted_V2(t *testing.T) {
	debug := false
	multithread := false
	poolsize := 1

	if multithread {
		poolsize = 16
	}
	fmt.Println("VCPUs: ", poolsize)

	sn := Loadcryptonet("cryptonet_packed.json")
	sn.Init()
	//crypto
	//8 levels neeeded
	ckksParams := ckks2.ParametersLiteral{
		LogN:     14,
		LogQ:     []int{35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, //Log(PQ) <= 438 for LogN 14
		LogP:     []int{37, 37, 37},
		Sigma:    rlwe2.DefaultSigma,
		LogSlots: 13,
		Scale:    float64(1 << 30),
	}

	params, err := ckks2.NewParametersFromLiteral(ckksParams)
	features := 784 //MNIST
	batchSize := 8
	alpha := 0.5 //ciphertext utilization factor. Set to 1 for max memory efficiency
	innerDim := int(math.Ceil((float64(params.N()) * alpha) / (2.0 * float64(batchSize))))
	innerDim = 8
	fmt.Printf("Input Dense: Rows %d, Cols %d --> InnerDim: %d\n", batchSize, features, innerDim)
	dataSn := data.LoadData("cryptonet_data_nopad.json")
	err = dataSn.Init(batchSize)

	X, Y, err := dataSn.Batch()
	Xpacked := PackBatchParallel(pu.NewDense(X), innerDim, params)

	weightMatrices, biasMatrices := sn.BuildParams(batchSize)
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
		PoolIdx:      0,
	}

	//convert to multidim packing
	snMD := sn.ConvertToMDPack(Xpacked.Batches(), innerDim, Xpacked.Rows(), Xpacked.Cols(), weightMatricesRescaled, biasMatricesRescaled, params, Box.Encoder, poolsize)

	//generate rotations and rot keys
	rotations := snMD.GenerateRotations(params)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)
	Box.Evaluator = ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rtks})

	batchEnc := md.NewBatchEncryptor(params, sk)

	maxIters := 5
	if multithread {
		maxIters = 1
	}

	poolElapsed := make([]float64, poolsize)
	poolAcc := make([]float64, poolsize)
	poolBatches := batchSize * poolsize

	var wg sync.WaitGroup
	for i := 0; i < poolsize; i++ {
		wg.Add(1)
		if i != 0 {
			X, Y, err = dataSn.Batch()
			if err != nil {
				continue
			}
			Xpacked = PackBatchParallel(pu.NewDense(X), innerDim, params)
		}
		Xenc := batchEnc.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Xpacked)
		fmt.Printf("Input dimentions:\nRows:%d\nCols:%d\nInnerDim:%d\nBatches:%d\n\n", Xenc.Rows(), Xenc.Cols(), Xenc.Dim(), Xpacked.Batches())

		go func(i int, X [][]float64, Y []int, Xpacked *md.PackedMatrix, Xenc *md.CiphertextBatchMatrix) {
			fmt.Printf("[!] Thread %d started\n", i+1)
			defer wg.Done()

			var elapsed int64
			tot := 0
			corrects := 0
			iters := 0

			var box md.Ckks2Box
			if i == 0 {
				box = Box
			} else {
				box = md.Ckks2Box{
					Params:       params,
					Encoder:      ckks2.NewEncoder(params),
					Evaluator:    ckks2.NewEvaluator(params, rlwe2.EvaluationKey{Rlk: rlk, Rtks: rtks}),
					Decryptor:    ckks2.NewDecryptor(params, sk),
					Encryptor:    ckks2.NewEncryptor(params, sk),
					BootStrapper: nil,
					PoolIdx:      i,
				}
			}
			snMD.InitPmMultiplier(params, box.Evaluator, box.PoolIdx)

			for true {
				var res utils.Stats
				if !debug {
					res = snMD.EvalBatchEncrypted(Y, Xenc, box)
				} else {
					res = snMD.EvalBatchEncrypted_Debug(Y, Xenc, box, pu.NewDense(X), weightMatrices, biasMatrices)
				}
				fmt.Println("Corrects/Tot:", res.Corrects, "/", batchSize)
				corrects += res.Corrects
				tot += batchSize
				elapsed += res.Time.Milliseconds()
				iters++
				X, Y, err = dataSn.Batch()
				Xpacked = PackBatchParallel(pu.NewDense(X), innerDim, params)
				Xenc = batchEnc.EncodeAndEncrypt(params.MaxLevel(), params.Scale(), Xpacked)
				if err != nil || iters >= maxIters {
					break
				}
			}
			poolElapsed[i] = float64(elapsed) / float64(iters)
			poolAcc[i] = float64(corrects) / float64(tot)
		}(i, X, Y, Xpacked, Xenc)
	}
	wg.Wait()
	avgElapsed := 0.0
	avgAcc := 0.0
	for i := range poolElapsed {
		avgElapsed += poolElapsed[i]
		avgAcc += poolAcc[i]
	}
	fmt.Println("TOT Samples: ", poolBatches)
	fmt.Println("TOT Latency (ms) :", float64(avgElapsed)/float64(poolsize))
	fmt.Println("TOT Accuracy :", float64(avgAcc)/float64(poolsize))
}
