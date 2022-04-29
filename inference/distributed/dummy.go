package distributed

import (
	"encoding/json"
	"errors"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"sync"
)

var TYPES = []string{"PubKeySwitch", "Refresh", "End"}

/*
	Dummy version of distributed protocols using channels for communication
*/
type DummyProtocol struct {
	Proto  interface{}   //a protocol type from dckks, keeps state (maybe add a finalizer function here?)
	shares []interface{} //collects shares from parties
}

type DummyProtocolMsg struct {
	Type string `json:"type"`
	Id   int    `json:"id"` //this is the id of the ct in the Enc Block Matrix, like i*row+j
	Dat  []byte `json:"dat"`
}
type DummyMaster struct {
	muxProtoBuf sync.RWMutex
	ProtoBuf    map[int]*DummyProtocol //ct id -> protocol instance
	sk          *rlwe.SecretKey
	Cpk         *rlwe.PublicKey
	Crs         *lattigoUtils.KeyedPRNG
	Params      ckks.Parameters
	Parties     int
	PlayerChans []chan []byte
}

type DummyPlayer struct {
	PCKS       *dckks.PCKSProtocol    //PubKeySwitch
	BTP        *dckks.RefreshProtocol //Bootstrap
	sk         *rlwe.SecretKey
	Cpk        *rlwe.PublicKey
	Crs        *lattigoUtils.KeyedPRNG
	Params     ckks.Parameters
	Id         int
	MasterChan chan []byte
}

func NewDummyMaster(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, crs *lattigoUtils.KeyedPRNG, parties int) (*DummyMaster, error) {
	master := new(DummyMaster)
	master.sk = sk
	master.Cpk = cpk
	master.Params = params
	master.Crs = crs
	master.Parties = parties

	master.PlayerChans = make([]chan []byte, parties-1)
	//master.PCKS = dckks.NewPCKSProtocol(params, 3.2)
	//var minLevel, logBound int
	//var ok bool
	//if minLevel, logBound, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q()); ok != true || minLevel+1 > params.MaxLevel() {
	//	return nil, -1, -1, errors.New("Not enough levels to ensure correcness and 128 security")
	//}
	//master.BTP = dckks.NewRefreshProtocol(params, logBound, 3.2)
	return master, nil
}
func NewDummyPlayer(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, crs *lattigoUtils.KeyedPRNG, id int, masterChan chan []byte) (*DummyPlayer, error) {
	player := new(DummyPlayer)
	player.sk = sk
	player.Cpk = cpk
	player.Params = params
	player.Crs = crs

	player.MasterChan = masterChan
	//master.PCKS = dckks.NewPCKSProtocol(params, 3.2)
	//var minLevel, logBound int
	//var ok bool
	//if minLevel, logBound, ok = dckks.GetMinimumLevelForBootstrapping(128, params.DefaultScale(), parties, params.Q()); ok != true || minLevel+1 > params.MaxLevel() {
	//	return nil, -1, -1, errors.New("Not enough levels to ensure correcness and 128 security")
	//}
	//master.BTP = dckks.NewRefreshProtocol(params, logBound, 3.2)
	return player, nil
}

func (dmst *DummyMaster) InitProto(proto interface{}, pkQ *rlwe.PublicKey, ct ckks.Ciphertext, ctId int) error {
	switch proto.(type) {
	case dckks.PCKSProtocol:
		//prepare PubKeySwitch
		go dmst.RunPubKeySwitch(pkQ, ct, ctId)
	case dckks.RefreshProtocol:
		//prepare Refresh
	default:
		return errors.New("Unknown protocol")
	}
	return nil
}

func (dmst *DummyMaster) RunPubKeySwitch(pkQ *rlwe.PublicKey, ct ckks.Ciphertext, ctId int) {
	dmst.muxProtoBuf.Lock()

	dmst.ProtoBuf[ctId] = &DummyProtocol{Proto: dckks.NewPCKSProtocol(dmst.Params, 3.2), shares: make([]interface{}, dmst.Parties)}
	proto := dmst.ProtoBuf[ctId].Proto.(dckks.PCKSProtocol)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(dmst.sk, pkQ, ct.Value[1], share)
	dmst.ProtoBuf[ctId].shares[0] = share

	dmst.muxProtoBuf.Unlock()

	dat, _ := ct.Value[1].MarshalBinary()
	msg := DummyProtocolMsg{Type: TYPES[0], Id: ctId, Dat: dat}
	buf, err := json.Marshal(msg)
	utils.ThrowErr(err)

	//"send" msgs
	for _, c := range dmst.PlayerChans {
		go func(c chan []byte) {
			c <- buf
		}(c)
	}
}

func (dp *DummyPlayer) Dispatch() {
	for {
		//listen from Master
		buf := <-dp.MasterChan
		var msg DummyProtocolMsg
		json.Unmarshal(buf, &msg)
		switch msg.Type {
		case TYPES[0]: //PubKeySwitch --> no need for memory, just execute and send
		case TYPES[1]: //Refresh
		case TYPES[2]: //End
		default:
			panic(errors.New("Unknown Protocol from Master"))
		}

	}
}
