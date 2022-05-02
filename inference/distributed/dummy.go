package distributed

import (
	"encoding/json"
	"errors"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/drlwe"
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"sync"
)

var TYPES = []string{"PubKeySwitch", "Refresh", "End"}

/*
	Dummy version of distributed protocols using channels for communication
*/
type DummyProtocol struct {
	muxProto     sync.RWMutex
	Proto        interface{}           //a protocol type from dckks, keeps state (maybe add a finalizer function here?)
	Shares       []interface{}         //collects shares from parties
	Completion   int                   //counter to completion
	Crp          drlwe.CKSCRP          //common reference polynomial if any
	FeedbackChan chan *ckks.Ciphertext //final result of protocol
}

//Extension for PCKS
type DummyPCKSExt struct {
	Pk []byte `json:"pk"` //Pub Key from Querier -> PubKeySwitch
}

//Extension for Refresh
type DummyRefreshExt struct {
	Crp       []byte  `json:"crp"`       //Common poly from CRS -> Refresh
	Precision int     `json:"precision"` //Precision for instance of Refresh Protocol
	MinLevel  int     `json:"minlevel"`
	Scale     float64 `json:"scale"`
}

type DummyProtocolMsg struct {
	Type string `json:"type"`
	Id   int    `json:"id"` //this is the id of the ct in the Enc Block Matrix, like i*row+j
	Ct   []byte `json:"ct"` //ciphertext

	//Protocol Dependent
	Extension interface{} `json:"extension"`
}

type DummyProtocolResp struct {
	ProtoId  int    `json:"protoId"`
	PlayerId int    `json:"playerId"`
	Share    []byte `json:"share"`
}

type DummyMaster struct {
	ProtoBuf    map[int]*DummyProtocol //ct id -> protocol instance (to be accessed in a parallel fashion)
	sk          *rlwe.SecretKey
	Cpk         *rlwe.PublicKey
	Crs         *lattigoUtils.KeyedPRNG //common reference string
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

//HELPERS

func MarshalCrp(crp drlwe.CKSCRP) ([]byte, error) {
	type V struct {
		Coeffs  [][]uint64 `json:"coeffs"`
		IsMForm bool       `json:"ismform"`
		IsNTT   bool       `json:"isntt"`
	}
	v := &V{Coeffs: crp.Coeffs, IsMForm: crp.IsMForm, IsNTT: crp.IsNTT}
	buf, err := json.Marshal(v)
	return buf, err
}

func UnMarshalCrp(buf []byte) (drlwe.CKSCRP, error) {
	type V struct {
		Coeffs  [][]uint64 `json:"coeffs"`
		IsMForm bool       `json:"ismform"`
		IsNTT   bool       `json:"isntt"`
	}
	v := new(V)
	err := json.Unmarshal(buf, v)
	var crp drlwe.CKSCRP
	crp.Coeffs = v.Coeffs
	crp.IsMForm = v.IsMForm
	crp.IsNTT = v.IsNTT
	return crp, err
}

func NewDummyMaster(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, crs *lattigoUtils.KeyedPRNG, parties int) (*DummyMaster, error) {
	master := new(DummyMaster)
	master.sk = sk
	master.Cpk = cpk
	master.Params = params
	master.Crs = crs
	master.Parties = parties
	//instance of protocol map
	master.ProtoBuf = make(map[int]*DummyProtocol)

	master.PlayerChans = make([]chan []byte, parties) //chan[0] is redundant
	return master, nil
}

func NewDummyPlayer(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, crs *lattigoUtils.KeyedPRNG, id int, masterChan chan []byte) (*DummyPlayer, error) {
	player := new(DummyPlayer)
	player.sk = sk
	player.Cpk = cpk
	player.Params = params
	player.Crs = crs
	player.Id = id

	player.MasterChan = masterChan
	return player, nil
}

//MASTER PROTOCOL

func (dmst *DummyMaster) InitProto(proto string, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) (*ckks.Ciphertext, error) {
	switch proto {
	case TYPES[0]:
		//prepare PubKeySwitch
		go dmst.RunPubKeySwitch(pkQ, ct, ctId)
		for _, c := range dmst.PlayerChans {
			go dmst.DispatchPCKS(c, ct)
		}
	case TYPES[1]:
		//prepare Refresh
		go dmst.RunRefresh(ct, ctId)
		for _, c := range dmst.PlayerChans {
			go dmst.DispatchRef(c, ct)
		}
	default:
		return nil, errors.New("Unknown protocol")
	}
	res := <-dmst.ProtoBuf[ctId].FeedbackChan
	return res, nil
}

//Initiates the PCKS protocol from master
func (dmst *DummyMaster) RunPubKeySwitch(pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) {
	//create protocol instance
	dmst.ProtoBuf[ctId] = &DummyProtocol{
		Proto:      dckks.NewPCKSProtocol(dmst.Params, 3.2),
		Shares:     make([]interface{}, dmst.Parties),
		Completion: 1,
	}
	proto := dmst.ProtoBuf[ctId].Proto.(dckks.PCKSProtocol)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(dmst.sk, pkQ, ct.Value[1], share)
	dmst.ProtoBuf[ctId].Shares[0] = share

	dat, _ := ct.Value[1].MarshalBinary()
	dat2, _ := pkQ.MarshalBinary()
	msg := DummyProtocolMsg{Type: TYPES[0], Id: ctId, Ct: dat}
	msg.Extension = DummyPCKSExt{Pk: dat2}
	buf, err := json.Marshal(msg)
	utils.ThrowErr(err)

	//"send" msgs
	for _, c := range dmst.PlayerChans {
		go func(c chan []byte) {
			c <- buf
		}(c)
	}
}

//Initiates the Refresh protocol from master
func (dmst *DummyMaster) RunRefresh(ct *ckks.Ciphertext, ctId int) error {
	//setup
	var minLevel, logBound int
	var ok bool
	if minLevel, logBound, ok = dckks.GetMinimumLevelForBootstrapping(128, ct.Scale, dmst.Parties, dmst.Params.Q()); ok != true || minLevel+1 > dmst.Params.MaxLevel() {
		return errors.New("Not enough levels to ensure correcness and 128 security")
	}
	//creates proto instance
	dmst.ProtoBuf[ctId] = &DummyProtocol{
		Proto:      dckks.NewRefreshProtocol(dmst.Params, logBound, 3.2),
		Shares:     make([]interface{}, dmst.Parties),
		Completion: 1,
	}
	proto := dmst.ProtoBuf[ctId].Proto.(dckks.RefreshProtocol)
	crp := proto.SampleCRP(dmst.Params.MaxLevel(), dmst.Crs)
	dmst.ProtoBuf[ctId].Crp = crp
	share := proto.AllocateShare(minLevel, dmst.Params.MaxLevel())
	proto.GenShare(dmst.sk, logBound, dmst.Params.LogSlots(), ct.Value[1], ct.Scale, crp, share)
	dmst.ProtoBuf[ctId].Shares[0] = share

	dat, _ := ct.Value[1].MarshalBinary()
	var dat2 []byte
	var err error
	if dat2, err = MarshalCrp(crp); err != nil {
		return err
	}
	msg := DummyProtocolMsg{Type: TYPES[1], Id: ctId, Ct: dat}
	msg.Extension = DummyRefreshExt{
		Crp:       dat2,
		Precision: logBound,
		MinLevel:  minLevel,
		Scale:     ct.Scale,
	}
	buf, err := json.Marshal(msg)
	utils.ThrowErr(err)

	//"send" msgs
	for _, c := range dmst.PlayerChans {
		go func(c chan []byte) {
			c <- buf
		}(c)
	}
	return nil
}

//Listen for shares and aggregates
func (dmst *DummyMaster) DispatchPCKS(c chan []byte, ct *ckks.Ciphertext) {
	for {
		buf := <-c
		var resp DummyProtocolResp
		err := json.Unmarshal(buf, &resp)
		utils.ThrowErr(err)

		dmst.ProtoBuf[resp.ProtoId].muxProto.Lock()

		var share drlwe.PCKSShare
		share.UnmarshalBinary(resp.Share)
		dmst.ProtoBuf[resp.ProtoId].Shares[resp.PlayerId] = share
		dmst.ProtoBuf[resp.ProtoId].Completion++
		if dmst.ProtoBuf[resp.ProtoId].Completion == dmst.Parties {
			//finalize
			proto := dmst.ProtoBuf[resp.ProtoId].Proto.(dckks.PCKSProtocol)
			for i := 1; i < dmst.Parties; i++ {
				proto.AggregateShare(
					dmst.ProtoBuf[resp.ProtoId].Shares[i].(*drlwe.PCKSShare),
					dmst.ProtoBuf[resp.ProtoId].Shares[0].(*drlwe.PCKSShare),
					dmst.ProtoBuf[resp.ProtoId].Shares[0].(*drlwe.PCKSShare))
			}
			ctSw := ckks.NewCiphertext(dmst.Params, 1, ct.Level(), ct.Scale)
			proto.KeySwitch(ct, dmst.ProtoBuf[resp.ProtoId].Shares[0].(*drlwe.PCKSShare), ctSw)
			dmst.ProtoBuf[resp.ProtoId].FeedbackChan <- ctSw
		}
		dmst.ProtoBuf[resp.ProtoId].muxProto.Unlock()
	}
}

//Listen for shares and Finalize
func (dmst *DummyMaster) DispatchRef(c chan []byte, ct *ckks.Ciphertext) *ckks.Ciphertext {
	for {
		buf := <-c
		var resp DummyProtocolResp
		err := json.Unmarshal(buf, &resp)
		utils.ThrowErr(err)
		dmst.ProtoBuf[resp.ProtoId].muxProto.Lock()
		var share dckks.RefreshShare
		share.UnmarshalBinary(resp.Share)
		dmst.ProtoBuf[resp.ProtoId].Shares[resp.PlayerId] = share
		dmst.ProtoBuf[resp.ProtoId].Completion++
		if dmst.ProtoBuf[resp.ProtoId].Completion == dmst.Parties {
			//finalize
			proto := dmst.ProtoBuf[resp.ProtoId].Proto.(dckks.RefreshProtocol)
			for i := 1; i < dmst.Parties; i++ {
				proto.AggregateShare(
					dmst.ProtoBuf[resp.ProtoId].Shares[i].(*dckks.RefreshShare),
					dmst.ProtoBuf[resp.ProtoId].Shares[0].(*dckks.RefreshShare),
					dmst.ProtoBuf[resp.ProtoId].Shares[0].(*dckks.RefreshShare))
			}
			ctFresh := ckks.NewCiphertext(dmst.Params, 1, dmst.Params.MaxLevel(), ct.Scale)
			proto.Finalize(ct, dmst.Params.LogSlots(), dmst.ProtoBuf[resp.ProtoId].Crp, dmst.ProtoBuf[resp.ProtoId].Shares[0].(*dckks.RefreshShare), ctFresh)
			dmst.ProtoBuf[resp.ProtoId].FeedbackChan <- ctFresh
		}
		dmst.ProtoBuf[resp.ProtoId].muxProto.Unlock()
	}
}

//PLAYERS PROTOCOL
func (dp *DummyPlayer) Dispatch() {
	for {
		//listen from Master
		buf := <-dp.MasterChan
		var msg DummyProtocolMsg
		json.Unmarshal(buf, &msg)
		switch msg.Type {
		case TYPES[0]: //PubKeySwitch --> no need for memory, just execute and send
			go dp.RunPubKeySwitch(msg)
		case TYPES[1]: //Refresh
			go dp.RunRefresh(msg)
		case TYPES[2]: //End
			go dp.End()
		default:
			panic(errors.New("Unknown Protocol from Master"))
		}

	}
}

//Generates and send share to Master
func (dp *DummyPlayer) RunPubKeySwitch(msg DummyProtocolMsg) {
	proto := dckks.NewPCKSProtocol(dp.Params, 3.2)
	var ct ring.Poly
	var pk rlwe.PublicKey
	Ext := msg.Extension.(DummyPCKSExt)
	err := ct.UnmarshalBinary(msg.Ct)
	utils.ThrowErr(err)
	err = pk.UnmarshalBinary(Ext.Pk)
	utils.ThrowErr(err)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(dp.sk, &pk, &ct, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := &DummyProtocolResp{Share: dat, PlayerId: dp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	dp.MasterChan <- dat
}

func (dp *DummyPlayer) RunRefresh(msg DummyProtocolMsg) {
	var ct ring.Poly

	ct.UnmarshalBinary(msg.Ct)
	msg.Extension = msg.Extension.(DummyRefreshExt)
	Ext := msg.Extension.(DummyRefreshExt)
	crp, err := UnMarshalCrp(Ext.Crp)
	precision := Ext.Precision
	scale := Ext.Scale
	minLevel := Ext.MinLevel

	proto := dckks.NewRefreshProtocol(dp.Params, precision, 3.2)
	share := proto.AllocateShare(minLevel, dp.Params.MaxLevel())
	proto.GenShare(dp.sk, precision, dp.Params.LogSlots(), &ct, scale, crp, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := &DummyProtocolResp{Share: dat, PlayerId: dp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	dp.MasterChan <- dat
}

func (dp *DummyPlayer) End() {
	close(dp.MasterChan)
}
