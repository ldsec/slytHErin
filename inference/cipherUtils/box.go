package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"os"
)

type RotationsSet struct {
	Set map[int]struct{}
}

func NewRotationsSet() *RotationsSet {
	rs := new(RotationsSet)
	rs.Set = make(map[int]struct{})
	return rs
}

func (rs *RotationsSet) Add(rots []int) {
	for i := range rots {
		if _, exists := rs.Set[rots[i]]; !exists {
			rs.Set[rots[i]] = struct{}{}
		}
	}
}

func (rs *RotationsSet) Rotations() []int {
	var rots []int
	for k, _ := range rs.Set {
		rots = append(rots, k)
	}
	return rots
}

//wrapper for the classes needed to perform encrypted operations, like a crypto-ToolBox
type CkksBox struct {
	Params       ckks.Parameters
	BtpParams    *bootstrapping.Parameters
	Encoder      ckks.Encoder
	Evaluator    ckks.Evaluator
	Encryptor    ckks.Encryptor
	Decryptor    ckks.Decryptor
	BootStrapper *bootstrapping.Bootstrapper
	Sk           *rlwe.SecretKey
	rtks         *rlwe.RotationKeySet
	evk          bootstrapping.EvaluationKeys
	kgen         ckks.KeyGenerator
}

func NewBox(params ckks.Parameters) CkksBox {
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	Box := CkksBox{
		Params:  params,
		Encoder: ckks.NewEncoder(params),
		Evaluator: ckks.NewEvaluator(params, rlwe.EvaluationKey{
			Rlk:  kgen.GenRelinearizationKey(sk, 2),
			Rtks: nil,
		}),
		Decryptor:    dec,
		Encryptor:    enc,
		BootStrapper: nil,
		Sk:           sk,
		kgen:         kgen,
	}
	return Box
}

func BoxShallowCopy(Box CkksBox) CkksBox {
	btp := new(bootstrapping.Bootstrapper)
	if Box.BootStrapper != nil {
		btp = Box.BootStrapper.ShallowCopy()
	}
	boxNew := CkksBox{
		Params:       Box.Params,
		Encoder:      Box.Encoder.ShallowCopy(),
		Evaluator:    Box.Evaluator.ShallowCopy(),
		Encryptor:    Box.Encryptor.ShallowCopy(),
		Decryptor:    Box.Decryptor.ShallowCopy(),
		BootStrapper: btp,
		Sk:           Box.Sk.CopyNew(),
		rtks:         Box.rtks,
		evk:          Box.evk,
		kgen:         Box.kgen,
	}
	return boxNew
}

//returns Box with Evaluator and Bootstrapper if needed
func BoxWithRotations(Box CkksBox, rotations []int, withBtp bool, btpParams *bootstrapping.Parameters) CkksBox {
	if rotations == nil {
		rotations = []int{}
	}
	if Box.rtks == nil {
		Box.rtks = Box.kgen.GenRotationKeysForRotations(rotations, true, Box.Sk)
	}
	params := Box.Params
	rlk := Box.kgen.GenRelinearizationKey(Box.Sk, 2)
	Box.Evaluator = ckks.NewEvaluator(Box.Params, rlwe.EvaluationKey{
		Rlk:  rlk,
		Rtks: Box.rtks,
	})
	var err error
	if withBtp {
		if btpParams != nil {
			rotations = btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
			Box.evk = bootstrapping.GenEvaluationKeys(*btpParams, Box.Params, Box.Sk)
			Box.BootStrapper, err = bootstrapping.NewBootstrapper(Box.Params, *btpParams, Box.evk)
			Box.BtpParams = btpParams
			utils.ThrowErr(err)
		} //else is distributed
	}
	return Box
}

//Generates rotatiosns needed for pipeline. Takes input features, as well inner rows,cols and partitions of weights as block matrices
/*
func GenRotations(rowIn, colIn, numWeights int, rowsW, colsW, rowPW, colPW []int, params ckks.Parameters, btpParams *bootstrapping.Parameters) []int {
	rotations := []int{}

	//implements a set
	rotSet := make(map[int]struct{})
	put := func(r int, rotations *[]int, rotSet *map[int]struct{}) {
		if _, exists := (*rotSet)[r]; !exists {
			*rotations = append(*rotations, r)
			(*rotSet)[r] = struct{}{}
		}
	}

	if btpParams != nil {
		rotations = btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
		for _, r := range rotations {
			rotSet[r] = struct{}{}
		}
	}
	var replicationFactor int
	currCols := colIn
	for w := 0; w < numWeights; w++ {
		for i := 1; i < (rowsW[w]+1)>>1; i++ {
			r := 2 * i * rowIn
			put(r, &rotations, &rotSet)
		}
		r := rowsW[w]
		put(r, &rotations, &rotSet)
		r = -rowsW[w] * rowIn
		put(r, &rotations, &rotSet)
		r = -2 * rowsW[w] * rowIn
		put(r, &rotations, &rotSet)

		if rowsW[w] < colsW[w] {
			replicationFactor = GetReplicaFactor(rowsW[w], colsW[w])
			R := params.RotationsForReplicateLog(rowIn*currCols, replicationFactor)
			for _, r := range R {
				put(r, &rotations, &rotSet)
			}
		}
		currCols = colsW[w]
		//check repack
		if w < numWeights-1 {
			if currCols != rowsW[w+1] {
				//repack
				R := GenRotationsForRepackCols(rowIn, currCols*colPW[w], currCols, rowPW[w+1])
				for _, r := range R {
					put(r, &rotations, &rotSet)
				}
				currCols = rowsW[w+1]
			}
		}
	}

	put(rowIn, &rotations, &rotSet)

	return rotations
}
*/

//serializes keys to disk
func SerializeBox(path string, Box CkksBox) {
	sk := Box.Sk
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
func DeserealizeBox(path string, params ckks.Parameters, btpParams *bootstrapping.Parameters, withBtp bool) CkksBox {
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
		BtpParams:    btpParams,
		Encoder:      ckks.NewEncoder(params),
		Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: kgen.GenRelinearizationKey(&sk, 2), Rtks: &rotKeys}),
		Encryptor:    ckks.NewEncryptor(params, &sk),
		Decryptor:    ckks.NewDecryptor(params, &sk),
		BootStrapper: nil,
		Sk:           &sk,
		rtks:         &rotKeys,
		evk:          bootstrapping.EvaluationKeys{},
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
		Box.BootStrapper, err = bootstrapping.NewBootstrapper(params, *btpParams, Box.evk)
		utils.ThrowErr(err)
	}
	return Box
}
