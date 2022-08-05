package distributed

import (
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/drlwe"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"net"
	"os"
	"strconv"
)

/*
	Collects all MPC protocols needed to produce keys which can be run in a setup phase,
	before interaction with the Querier
	Thus, they are implemented in a dummy way
*/

type SetupMsg struct {
	SkShare *rlwe.SecretKey `json:"skShare,omitempty"`
	Pk      *rlwe.PublicKey `json:"pk,omitempty"`
	Id      int             `json:"id,omitempty"`
	Addr    string          `json:"addr,omitempty"`
}

//Creates a player on remote node after receiving setup parameters from master. It is a dummy setup
func Setup(addr string, params ckks.Parameters) *LocalPlayer {
	addrTCP, err := net.ResolveTCPAddr("tcp", addr)
	utils.ThrowErr(err)
	listener, err := net.ListenTCP("tcp", addrTCP)
	utils.ThrowErr(err)
	conn, err := listener.AcceptTCP()
	utils.ThrowErr(err)

	data, err := ReadFrom(conn)
	utils.ThrowErr(err)
	var setupMsg SetupMsg
	err = json.Unmarshal(data, &setupMsg)
	utils.ThrowErr(err)
	player, err := NewLocalPlayer(setupMsg.SkShare, setupMsg.Pk, params, setupMsg.Id, setupMsg.Addr)
	return player
}

//Returns array of secret key shares, secret key and collective encryption key
func DummyEncKeyGen(params ckks.Parameters, crs *lattigoUtils.KeyedPRNG, parties int) ([]*rlwe.SecretKey, *rlwe.SecretKey, *rlwe.PublicKey, ckks.KeyGenerator) {
	type Party struct {
		*dckks.CKGProtocol
		sk *rlwe.SecretKey
		p  *drlwe.CKGShare //share disclosed
	}
	kgen := ckks.NewKeyGenerator(params)
	skShares := make([]*rlwe.SecretKey, parties)
	sk := ckks.NewSecretKey(params)
	ringQP, levelQ, levelP := params.RingQP(), params.QCount()-1, params.PCount()-1
	for i := 0; i < parties; i++ {
		skShares[i] = kgen.GenSecretKey()
		ringQP.AddLvl(levelQ, levelP, sk.Value, skShares[i].Value, sk.Value)
	}

	ckgParties := make([]*Party, parties)
	for i := 0; i < parties; i++ {
		p := new(Party)
		p.CKGProtocol = dckks.NewCKGProtocol(params)
		p.sk = skShares[i]
		p.p = p.AllocateShare()
		ckgParties[i] = p
	}
	P0 := ckgParties[0]

	crp := P0.SampleCRP(crs) //common reference poly

	// Each party creates a new CKGProtocol instance
	for i, p := range ckgParties {
		p.GenShare(p.sk, crp, p.p)
		if i > 0 {
			P0.AggregateShare(p.p, P0.p, P0.p)
		}
	}

	pk := ckks.NewPublicKey(params)
	P0.GenPublicKey(P0.p, crp, pk)
	return skShares, sk, pk, kgen
}

func DummyRelinKeyGen(params ckks.Parameters, crs *lattigoUtils.KeyedPRNG, shares []*rlwe.SecretKey) *rlwe.RelinearizationKey {
	type Party struct {
		*dckks.RKGProtocol
		ephSk  *rlwe.SecretKey //ephemeral secret used in rounds
		sk     *rlwe.SecretKey
		share1 *drlwe.RKGShare //round 1 disclosed share
		share2 *drlwe.RKGShare //round 2...
	}
	parties := len(shares)

	rkgParties := make([]*Party, parties)
	for i := range rkgParties {
		p := new(Party)
		p.RKGProtocol = dckks.NewRKGProtocol(params)
		p.sk = shares[i]
		p.ephSk, p.share1, p.share2 = p.AllocateShare()
		rkgParties[i] = p
	}
	P0 := rkgParties[0]

	crp := P0.SampleCRP(crs)

	// ROUND 1
	for i, p := range rkgParties {
		p.GenShareRoundOne(p.sk, crp, p.ephSk, p.share1)
		if i > 0 {
			P0.AggregateShare(p.share1, P0.share1, P0.share1)
		}
	}

	//ROUND 2
	for i, p := range rkgParties {
		p.GenShareRoundTwo(p.ephSk, p.sk, P0.share1, p.share2)
		if i > 0 {
			P0.AggregateShare(p.share2, P0.share2, P0.share2)
		}
	}
	rlk := ckks.NewRelinearizationKey(params)
	P0.GenRelinearizationKey(P0.share1, P0.share2, rlk)
	return rlk
}

func SerializeKeys(sk *rlwe.SecretKey, skshares []*rlwe.SecretKey, rtks *rlwe.RotationKeySet, path string) {
	fmt.Println("Writing keys to disk: ", path)
	dat, err := sk.MarshalBinary()
	utils.ThrowErr(err)
	f, err := os.Create(path + "_sk")
	utils.ThrowErr(err)
	_, err = f.Write(dat)
	utils.ThrowErr(err)
	f.Close()

	for i, sks := range skshares {
		dat, err := sks.MarshalBinary()
		utils.ThrowErr(err)
		f, err := os.Create(path + "_P" + strconv.Itoa(i+1) + "_sk")
		utils.ThrowErr(err)
		_, err = f.Write(dat)
		utils.ThrowErr(err)
		f.Close()
	}

	dat, err = rtks.MarshalBinary()
	utils.ThrowErr(err)
	f, err = os.Create(path + "_rtks")
	utils.ThrowErr(err)
	_, err = f.Write(dat)
	utils.ThrowErr(err)
	f.Close()
}

func DeserializeKeys(path string, parties int) (sk *rlwe.SecretKey, skShares []*rlwe.SecretKey, rtks *rlwe.RotationKeySet) {
	sk = new(rlwe.SecretKey)
	rtks = new(rlwe.RotationKeySet)
	skShares = make([]*rlwe.SecretKey, parties)
	for i := range skShares {
		skShares[i] = new(rlwe.SecretKey)
	}
	fmt.Println("Reading keys from disk: ", path)
	dat, err := os.ReadFile(path + "_sk")
	utils.ThrowErr(err)
	sk.UnmarshalBinary(dat)

	dat, err = os.ReadFile(path + "_rtks")
	utils.ThrowErr(err)
	rtks.UnmarshalBinary(dat)

	for i := 0; i < parties; i++ {
		dat, err := os.ReadFile(path + "_P" + strconv.Itoa(i+1) + "_sk")
		utils.ThrowErr(err)
		skShares[i].UnmarshalBinary(dat)
	}
	return
}
