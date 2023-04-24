package distributed

import (
	"fmt"
	"github.com/ldsec/slytHErin/inference/cipherUtils"
	pU "github.com/ldsec/slytHErin/inference/plainUtils"
	"github.com/ldsec/slytHErin/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"strconv"
	"testing"
	"time"
)

func TestBootstrapDistributed(t *testing.T) {
	PARTIES := []int{3, 5, 10}
	paramsCentralized, _ := ckks.NewParametersFromLiteral(bootstrapping.N15QP768H192H32.SchemeParams)
	paramsOptimized, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:         14,
		LogSlots:     13,
		LogQ:         []int{40, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31},
		LogP:         []int{55},
		DefaultScale: 1 << 31,
		Sigma:        rlwe.DefaultSigma,
		RingType:     ring.Standard,
	})
	PARAMS := []ckks.Parameters{paramsCentralized, paramsOptimized}
	L := pU.RandMatrix(64, 64)
	L.Set(0, 0, 30)
	for _, params := range PARAMS {
		for _, parties := range PARTIES { //parties = 3 is fine always
			fmt.Printf("Test: parties %d, params %d \n\n", parties, params.LogN())
			crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'R', 'A', 'N', 'D'})
			skShares, skP, pkP, _ := DummyEncKeyGen(params, crs, parties)
			decP := ckks.NewDecryptor(params, skP)
			rlk := DummyRelinKeyGen(params, crs, skShares)
			Box := cipherUtils.CkksBox{
				Params:       params,
				Encoder:      ckks.NewEncoder(params),                                            //public
				Evaluator:    ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: nil}), //from parties
				Decryptor:    decP,                                                               //from parties for debug
				Encryptor:    ckks.NewEncryptor(params, pkP),                                     //from parties
				BootStrapper: nil,
			}
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
			//[!] Start distributed parties
			master, err := NewLocalMaster(skShares[0], pkP, params, parties, partiesAddr, 1, true)
			utils.ThrowErr(err)
			players := make([]*LocalPlayer, parties-1)
			for i := 0; i < parties-1; i++ {
				players[i], err = NewLocalPlayer(skShares[i+1], pkP, params, i+1, partiesAddr[i+1], true)
				go players[i].Listen()
				utils.ThrowErr(err)
			}
			minLevel, _, _ := dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q())

			Btp := NewDistributedBootstrapper(master, 1)

			ct, _ := cipherUtils.NewEncInput(L, 1, 1, minLevel, params.DefaultScale(), Box)
			start := time.Now()
			Btp.Bootstrap(ct, Box)
			utils.ThrowErr(err)
			blocks, _ := pU.PartitionMatrix(L, 1, 1)
			cipherUtils.PrintDebugBlocks(ct, blocks, 0.1, Box)
			fmt.Printf("Test: parties %d, params %d\n\n", parties, params.LogN())
			fmt.Println("End: ", time.Since(start).Seconds())
			master.StartProto(END, nil, nil, 0, Box)
			time.Sleep(5000 * time.Millisecond) //wait for stop
		}
	}
}
