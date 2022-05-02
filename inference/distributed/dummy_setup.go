package distributed

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/drlwe"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
)

/*
	Collects all MPC protocols needed to produce keys which can be run in a setup phase,
	before interaction with the Querier
	Thus, they are implemented in a dummy way
*/

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
