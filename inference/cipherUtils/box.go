package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	utils2 "github.com/ldsec/lattigo/v2/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"math"
	"os"
)

//wrapper for the classes needed to perform encrypted operations, like a crypto-ToolBox
type CkksBox struct {
	Params       ckks.Parameters
	Encoder      ckks.Encoder
	Evaluator    ckks.Evaluator
	Encryptor    ckks.Encryptor
	Decryptor    ckks.Decryptor
	BootStrapper *bootstrapping.Bootstrapper
	sk           *rlwe.SecretKey
	rtks         *rlwe.RotationKeySet
	evk          bootstrapping.EvaluationKeys
	kgen         ckks.KeyGenerator
}

func NewBox(params ckks.Parameters) CkksBox {
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	//init rotations
	//rotations are performed between submatrixes

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	Box := CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: kgen.GenRelinearizationKey(sk, 2)}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
		sk:           sk,
		kgen:         kgen,
	}
	return Box
}

func BoxShallowCopy(Box CkksBox) CkksBox {
	boxNew := CkksBox{
		Params:    Box.Params,
		Encoder:   Box.Encoder.ShallowCopy(),
		Evaluator: Box.Evaluator.ShallowCopy(),
		Decryptor: Box.Decryptor.ShallowCopy(),
		Encryptor: Box.Encryptor.ShallowCopy(),
	}
	return boxNew
}

//returns Box with Evaluator and Bootstrapper if needed
func BoxWithSplits(Box CkksBox, btpParams bootstrapping.Parameters, withBtp bool, splits []BlockSplits) CkksBox {
	info, _ := ExctractInfo(splits)
	rotations := GenRotations(info.InputRows, info.InputCols, info.NumWeights, info.RowsOfWeights, info.ColsOfWeights, info.RowPOfWeights, info.ColPOfWeights, Box.Params, &btpParams)

	rlk := Box.kgen.GenRelinearizationKey(Box.sk, 2)
	Box.rtks = Box.kgen.GenRotationKeysForRotations(rotations, true, Box.sk)
	Box.Evaluator = ckks.NewEvaluator(Box.Params, rlwe.EvaluationKey{
		Rlk:  rlk,
		Rtks: Box.rtks,
	})
	var err error
	if withBtp {
		Box.evk = bootstrapping.GenEvaluationKeys(btpParams, Box.Params, Box.sk)
		Box.BootStrapper, err = bootstrapping.NewBootstrapper(Box.Params, btpParams, Box.evk)
		utils.ThrowErr(err)
	}
	return Box
}

//returns Box with Evaluator and Bootstrapper if needed
func BoxWithRotations(Box CkksBox, rotations []int, withBtp bool, btpParams bootstrapping.Parameters) CkksBox {

	rlk := Box.kgen.GenRelinearizationKey(Box.sk, 2)
	Box.rtks = Box.kgen.GenRotationKeysForRotations(rotations, true, Box.sk)
	Box.Evaluator = ckks.NewEvaluator(Box.Params, rlwe.EvaluationKey{
		Rlk:  rlk,
		Rtks: Box.rtks,
	})
	var err error
	if withBtp {
		Box.evk = bootstrapping.GenEvaluationKeys(btpParams, Box.Params, Box.sk)
		Box.BootStrapper, err = bootstrapping.NewBootstrapper(Box.Params, btpParams, Box.evk)
		utils.ThrowErr(err)
	}
	return Box
}

//Generates rotatiosns needed for pipeline. Takes input features, as well inner rows,cols and partitions of weights as block matrices
func GenRotations(rowIn, colIn, numWeights int, rowsW, colsW, rowPW, colPW []int, params ckks.Parameters, btpParams *bootstrapping.Parameters) []int {
	rotations := []int{}
	if btpParams != nil {
		rotations = btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
	}
	var replicationFactor int
	currCols := colIn
	for w := 0; w < numWeights; w++ {
		for i := 1; i < (rowsW[w]+1)>>1; i++ {
			rotations = append(rotations, 2*i*rowIn)
		}
		rotations = append(rotations, rowsW[w])
		rotations = append(rotations, -rowsW[w]*rowIn)
		rotations = append(rotations, -2*rowsW[w]*rowIn)
		if rowsW[w] < colsW[w] {
			replicationFactor = plainUtils.Max(int(math.Ceil(float64(colsW[w]/rowsW[w]))), 3)
			rotations = append(rotations, params.RotationsForReplicateLog(rowIn*currCols, replicationFactor)...)
		}
		currCols = colsW[w]
		//check repack
		if w < numWeights-1 {
			if currCols != rowsW[w+1] {
				//repack
				rotations = append(rotations, GenRotationsForRepackCols(rowIn, currCols*colPW[w], currCols, rowPW[w+1])...)
				currCols = rowsW[w+1]
			}
		}
	}

	rotations = append(rotations, rowIn)

	return rotations
}

// GenSubVectorRotationMatrix allows to generate a permutation matrix that roates subvectors independently.
// Given a vector of size N=2^"logSlots", partitionned into N/"vectorSize" subvectors each of size "vectorSize",
// rotates each subvector by "k" positions to the left.
//
// Example :
// Given v = [a_(0), a_(1), a_(2), ..., a_(N-3), a_(N-2), a_(N-1)],
// Then M x v = [rotate(a_(0), a_(1), ..., a_(vectorsize-1), k), ... , rotate(a_(N-vectorsize-1), a_(N-vectorsize), ..., a_(N-1), k)]
//
// If vectorSize does not divide N, then the last N%vectorSize slots are zero.
// If N = vectorSize, then no mask is generated and the evaluation is instead a single rotation.
//
// This is done by generating the two masks :
//       	 |     vectorsize     |, ..., |     vectorsize     |
// mask_0 = [{1, ..., 1, 0, ..., 0}, ..., {1, ..., 1, 0, ..., 0}]
// mask_1 = [{0, ..., 0, 1, ..., 1}, ..., {0, ..., 0, 1, ..., 1}]
//            0 ----- k                    0 ----- k
func GenSubVectorRotationMatrix(params ckks.Parameters, level int, scale float64, vectorSize, k int, logSlots int, encoder ckks.Encoder) (matrix ckks.LinearTransform) {

	k %= vectorSize

	diagMatrix := make(map[int][]complex128)

	slots := 1 << logSlots

	matrix.Vec = make(map[int]rlwe.PolyQP)

	if vectorSize < slots {
		m0 := make([]complex128, slots)
		m1 := make([]complex128, slots)

		for i := 0; i < slots/vectorSize; i++ {

			index := i * vectorSize

			for j := 0; j < k; j++ {
				m0[j+index] = 1
			}

			for j := k; j < vectorSize; j++ {
				m1[j+index] = 1
			}
		}

		diagMatrix[slots-vectorSize+k] = m0
		diagMatrix[k] = m1

		// Encoding
		matrix = ckks.NewLinearTransform(params, []int{slots - vectorSize + k, k}, level, params.LogSlots(), 0)
		matrix.LogSlots = logSlots
		matrix.Level = level
		matrix.Scale = scale
		// Encode m0
		encoder.Embed(utils2.RotateComplex128Slice(m0, slots-vectorSize+k), params.LogSlots(), scale, true, matrix.Vec[slots-vectorSize+k])

		// Encode m1
		encoder.Embed(utils2.RotateComplex128Slice(m1, k), params.LogSlots(), scale, true, matrix.Vec[k])

	} else {
		// If N = vectorSize, the we a single rotation without masking is sufficient
		matrix.Vec[k] = rlwe.PolyQP{}
	}

	return matrix
}

//serializes keys to disk
func SerializeBox(path string, Box CkksBox) {
	sk := Box.sk
	rotKeys := Box.rtks
	fmt.Println("Writing keys to disk: ", path)
	dat, err := sk.MarshalBinary()
	utils.ThrowErr(err)
	f, err := os.Create(path + "_sk")
	utils.ThrowErr(err)
	_, err = f.Write(dat)
	utils.ThrowErr(err)
	f.Close()

	dat, err = rotKeys.MarshalBinary()
	utils.ThrowErr(err)
	f, err = os.Create(path + "_rtks")
	utils.ThrowErr(err)
	_, err = f.Write(dat)
	utils.ThrowErr(err)
	f.Close()

	if Box.BootStrapper != nil {
		dat, err = Box.evk.Rlk.MarshalBinary()
		utils.ThrowErr(err)
		f, err = os.Create(path + "_btp_rlk")
		utils.ThrowErr(err)
		_, err = f.Write(dat)
		utils.ThrowErr(err)
		f.Close()

		dat, err = Box.evk.Rtks.MarshalBinary()
		utils.ThrowErr(err)
		f, err = os.Create(path + "_btp_rtks")
		utils.ThrowErr(err)
		_, err = f.Write(dat)
		utils.ThrowErr(err)
		f.Close()

		dat, err = Box.evk.SwkDtS.MarshalBinary()
		utils.ThrowErr(err)
		f, err = os.Create(path + "_btp_swkDtS")
		utils.ThrowErr(err)
		_, err = f.Write(dat)
		utils.ThrowErr(err)
		f.Close()

		dat, err = Box.evk.SwkStD.MarshalBinary()
		utils.ThrowErr(err)
		f, err = os.Create(path + "_btp_swkStD")
		utils.ThrowErr(err)
		_, err = f.Write(dat)
		utils.ThrowErr(err)
		f.Close()
	}
}

//loads serialized keys from disk into a fresh box
func DeserealizeBox(path string, params ckks.Parameters, btpParams bootstrapping.Parameters, withBtp bool) CkksBox {
	fmt.Println("Reading keys from disk: ", path)
	dat, err := os.ReadFile(path + "_sk")
	utils.ThrowErr(err)
	var sk rlwe.SecretKey
	sk.UnmarshalBinary(dat)

	dat, err = os.ReadFile(path + "_rtks")
	utils.ThrowErr(err)
	var rotKeys rlwe.RotationKeySet
	rotKeys.UnmarshalBinary(dat)

	kgen := ckks.NewKeyGenerator(params)

	Box := CkksBox{
		Params:       params,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: kgen.GenRelinearizationKey(&sk, 2), Rtks: &rotKeys}),
		Decryptor:    ckks.NewDecryptor(params, &sk),
		Encryptor:    ckks.NewEncryptor(params, &sk),
		BootStrapper: nil,
		sk:           &sk,
		rtks:         &rotKeys,
		kgen:         kgen,
	}
	if withBtp {
		fmt.Println("Reading btp keys")
		dat, err := os.ReadFile(path + "_btp_rlk")
		utils.ThrowErr(err)
		var rlk rlwe.RelinearizationKey
		rlk.UnmarshalBinary(dat)

		dat, err = os.ReadFile(path + "_btp_rtks")
		utils.ThrowErr(err)
		var btpRotKeys rlwe.RotationKeySet
		btpRotKeys.UnmarshalBinary(dat)

		dat, err = os.ReadFile(path + "_btp_swkDtS")
		utils.ThrowErr(err)
		var keySwDtS rlwe.SwitchingKey
		keySwDtS.UnmarshalBinary(dat)

		dat, err = os.ReadFile(path + "_btp_swkStD")
		utils.ThrowErr(err)
		var keySwStD rlwe.SwitchingKey
		keySwStD.UnmarshalBinary(dat)

		Box.evk = bootstrapping.EvaluationKeys{
			EvaluationKey: rlwe.EvaluationKey{Rlk: &rlk, Rtks: &btpRotKeys},
			SwkDtS:        &keySwDtS,
			SwkStD:        &keySwStD,
		}
		Box.BootStrapper, err = bootstrapping.NewBootstrapper(params, btpParams, Box.evk)
		utils.ThrowErr(err)
	}
	return Box
}
