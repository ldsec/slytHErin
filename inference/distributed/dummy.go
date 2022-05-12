package distributed

import (
	"encoding/json"
	"errors"
	"fmt"
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
	Protocol     interface{}      //instance of protocol
	Crp          drlwe.CKSCRP     //Common reference poly if any
	Ct           *ckks.Ciphertext //ciphertext of the protocol
	muxProto     sync.RWMutex
	Shares       []interface{}         //collects shares from parties
	Completion   int                   //counter to completion
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
	Type     string `json:"type"`
	PlayerId int    `json:"playerId"`
	Share    []byte `json:"share"`
}

type DummyMaster struct {
	ProtoBuf *sync.Map //ct id -> protocol instance *DummyProtocol
	sk       *rlwe.SecretKey
	Cpk      *rlwe.PublicKey
	Params   ckks.Parameters
	Parties  int
	//full duplex comm
	M2PChans []chan []byte //master to players
	P2MChans []chan []byte //players to master

	//for Ending
	runningMux    sync.RWMutex
	runningProtos int       //counter for how many protos are running
	Done          chan bool //flag caller that master is done with all instances
}

type DummyPlayer struct {
	PCKS           *dckks.PCKSProtocol    //PubKeySwitch
	BTP            *dckks.RefreshProtocol //Bootstrap
	sk             *rlwe.SecretKey
	Cpk            *rlwe.PublicKey
	Params         ckks.Parameters
	Id             int
	ToMasterChan   chan []byte
	FromMasterChan chan []byte
}

type Handler func(c chan []byte)

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

func NewDummyMaster(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, parties int) (*DummyMaster, error) {
	master := new(DummyMaster)
	master.sk = sk
	master.Cpk = cpk
	master.Params = params
	master.Parties = parties
	//instance of protocol map
	master.ProtoBuf = new(sync.Map)

	master.M2PChans = make([]chan []byte, parties) //chan[0] is redundant
	master.P2MChans = make([]chan []byte, parties)
	for i := 0; i < parties; i++ {
		master.M2PChans[i] = make(chan []byte)
		master.P2MChans[i] = make(chan []byte)
	}
	go close(master.M2PChans[0])
	go close(master.P2MChans[0])
	return master, nil
}

func NewDummyPlayer(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, id int, ToMasterChan chan []byte, FromMasterChan chan []byte) (*DummyPlayer, error) {
	player := new(DummyPlayer)
	player.sk = sk
	player.Cpk = cpk
	player.Params = params
	player.Id = id

	player.ToMasterChan = ToMasterChan
	player.FromMasterChan = FromMasterChan
	return player, nil
}

//MASTER PROTOCOL
func (dmst *DummyMaster) InitProto(proto string, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) (*ckks.Ciphertext, error) {
	switch proto {
	case TYPES[0]:
		//prepare PubKeySwitch
		fmt.Printf("[*] Master -- Registering PubKeySwitch ID: %d\n\n", ctId)
		protocol := dckks.NewPCKSProtocol(dmst.Params, 3.2)
		dmst.ProtoBuf.Store(ctId, &DummyProtocol{
			Protocol:     protocol,
			Ct:           ct,
			Shares:       make([]interface{}, dmst.Parties),
			Completion:   0,
			FeedbackChan: make(chan *ckks.Ciphertext),
		})
		go dmst.RunPubKeySwitch(protocol.ShallowCopy(), pkQ, ct, ctId)

	case TYPES[1]:
		//prepare Refresh
		fmt.Printf("[*] Master -- Registering Refresh ID: %d\n\n", ctId)
		var minLevel, logBound int
		var ok bool
		if minLevel, logBound, ok = dckks.GetMinimumLevelForBootstrapping(128, ct.Scale, dmst.Parties, dmst.Params.Q()); ok != true || minLevel+1 > dmst.Params.MaxLevel() {
			utils.ThrowErr(errors.New("Not enough levels to ensure correctness and 128 security"))
		}
		//creates proto instance
		protocol := dckks.NewRefreshProtocol(dmst.Params, logBound, 3.2)
		crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})
		crp := protocol.SampleCRP(dmst.Params.MaxLevel(), crs)
		dmst.ProtoBuf.Store(ctId, &DummyProtocol{
			Protocol:     protocol,
			Ct:           ct,
			Crp:          crp,
			Shares:       make([]interface{}, dmst.Parties),
			Completion:   0,
			FeedbackChan: make(chan *ckks.Ciphertext),
		})
		go dmst.RunRefresh(protocol.ShallowCopy(), ct, crp, minLevel, logBound, ctId)
	default:
		return nil, errors.New("Unknown protocol")
	}
	entry, _ := dmst.ProtoBuf.Load(ctId)
	res := <-entry.(*DummyProtocol).FeedbackChan
	dmst.ProtoBuf.Delete(ctId)
	return res, nil
}

func (dmst *DummyMaster) Listen() {
	for i := 0; i < dmst.Parties-1; i++ {
		fmt.Println("[*] Master listening on chan: ", i+1)
		go dmst.Dispatch(dmst.P2MChans[i+1])
	}
}

func (dmst *DummyMaster) Dispatch(c chan []byte) {
	for {
		buf := <-c
		var resp DummyProtocolResp
		err := json.Unmarshal(buf, &resp)
		utils.ThrowErr(err)
		switch resp.Type {
		case TYPES[0]:
			//key switch
			go dmst.DispatchPCKS(resp)
		case TYPES[1]:
			go dmst.DispatchRef(resp)
		default:
			utils.ThrowErr(errors.New("resp for unknown protocol"))
		}
	}
}

//Initiates the PCKS protocol from master
func (dmst *DummyMaster) RunPubKeySwitch(proto *dckks.PCKSProtocol, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) {
	//create protocol instance
	entry, _ := dmst.ProtoBuf.Load(ctId)
	entry.(*DummyProtocol).muxProto.Lock()
	//proto := entry.(*DummyProtocol).Proto.(*dckks.PCKSProtocol)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(dmst.sk, pkQ, ct.Value[1], share)
	entry.(*DummyProtocol).Shares[0] = share
	entry.(*DummyProtocol).Completion++
	entry.(*DummyProtocol).muxProto.Unlock()
	dmst.ProtoBuf.Store(ctId, entry.(*DummyProtocol))

	dat, _ := ct.Value[1].MarshalBinary()
	dat2, _ := pkQ.MarshalBinary()
	msg := DummyProtocolMsg{Type: TYPES[0], Id: ctId, Ct: dat}
	msg.Extension = DummyPCKSExt{Pk: dat2}
	buf, err := json.Marshal(msg)
	fmt.Println("Len proto message: ", len(dat))
	utils.ThrowErr(err)

	//"send" msgs
	for i := 0; i < dmst.Parties-1; i++ {
		c := dmst.M2PChans[i+1]
		go func(c chan []byte) {
			c <- buf
		}(c)
	}
}

//Initiates the Refresh protocol from master
func (dmst *DummyMaster) RunRefresh(proto *dckks.RefreshProtocol, ct *ckks.Ciphertext, crp drlwe.CKSCRP, minLevel int, logBound int, ctId int) {
	//setup
	entry, _ := dmst.ProtoBuf.Load(ctId)
	share := proto.AllocateShare(minLevel, dmst.Params.MaxLevel())
	proto.GenShare(dmst.sk, logBound, dmst.Params.LogSlots(), ct.Value[1], ct.Scale, crp, share)
	entry.(*DummyProtocol).Shares[0] = share
	entry.(*DummyProtocol).Completion++
	dmst.ProtoBuf.Store(ctId, entry.(*DummyProtocol))
	dat, _ := ct.Value[1].MarshalBinary()
	var dat2 []byte
	var err error
	if dat2, err = MarshalCrp(crp); err != nil {
		utils.ThrowErr(err)
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
	fmt.Println("Len proto message: ", len(dat))

	//"send" msgs
	for i := 0; i < dmst.Parties-1; i++ {
		c := dmst.M2PChans[i+1]
		go func(c chan []byte) {
			c <- buf
		}(c)
	}
}

//Listen for shares and aggregates
func (dmst *DummyMaster) DispatchPCKS(resp DummyProtocolResp) {
	if resp.PlayerId == 0 {
		//ignore invalid
		return
	}
	fmt.Printf("[*] Master -- Received share of PCKS ID: %d from %d\n\n", resp.ProtoId, resp.PlayerId)
	entry, ok := dmst.ProtoBuf.Load(resp.ProtoId)
	if !ok {
		return
	}
	proto := entry.(*DummyProtocol).Protocol.(*dckks.PCKSProtocol)
	ct := entry.(*DummyProtocol).Ct
	entry.(*DummyProtocol).muxProto.Lock()

	share := new(drlwe.PCKSShare)
	share.UnmarshalBinary(resp.Share)
	entry.(*DummyProtocol).Shares[resp.PlayerId] = share
	entry.(*DummyProtocol).Completion++
	if entry.(*DummyProtocol).Completion == dmst.Parties {
		//finalize
		for i := 1; i < dmst.Parties; i++ {
			proto.AggregateShare(
				entry.(*DummyProtocol).Shares[i].(*drlwe.PCKSShare),
				entry.(*DummyProtocol).Shares[0].(*drlwe.PCKSShare),
				entry.(*DummyProtocol).Shares[0].(*drlwe.PCKSShare))
		}
		ctSw := ckks.NewCiphertext(dmst.Params, 1, ct.Level(), ct.Scale)
		proto.KeySwitch(ct, entry.(*DummyProtocol).Shares[0].(*drlwe.PCKSShare), ctSw)
		entry.(*DummyProtocol).FeedbackChan <- ctSw
	}
	entry.(*DummyProtocol).muxProto.Unlock()
	dmst.ProtoBuf.Store(resp.ProtoId, entry.(*DummyProtocol))
}

//Listen for shares and Finalize
func (dmst *DummyMaster) DispatchRef(resp DummyProtocolResp) {
	if resp.PlayerId == 0 {
		//ignore invalid
		return
	}
	fmt.Printf("[*] Master -- Received share of Refresh ID: %d from %d\n\n", resp.ProtoId, resp.PlayerId)
	entry, ok := dmst.ProtoBuf.Load(resp.ProtoId)
	if !ok {
		return
	}
	proto := entry.(*DummyProtocol).Protocol.(*dckks.RefreshProtocol)
	crp := entry.(*DummyProtocol).Crp
	ct := entry.(*DummyProtocol).Ct
	entry.(*DummyProtocol).muxProto.Lock()
	share := new(dckks.RefreshShare)
	share.UnmarshalBinary(resp.Share)
	entry.(*DummyProtocol).Shares[resp.PlayerId] = share
	entry.(*DummyProtocol).Completion++
	if entry.(*DummyProtocol).Completion == dmst.Parties {
		//finalize
		for i := 1; i < dmst.Parties; i++ {
			proto.AggregateShare(
				entry.(*DummyProtocol).Shares[i].(*dckks.RefreshShare),
				entry.(*DummyProtocol).Shares[0].(*dckks.RefreshShare),
				entry.(*DummyProtocol).Shares[0].(*dckks.RefreshShare))
		}
		ctFresh := ckks.NewCiphertext(dmst.Params, 1, dmst.Params.MaxLevel(), ct.Scale)
		proto.Finalize(ct, dmst.Params.LogSlots(), crp, entry.(*DummyProtocol).Shares[0].(*dckks.RefreshShare), ctFresh)
		entry.(*DummyProtocol).FeedbackChan <- ctFresh
	}
	entry.(*DummyProtocol).muxProto.Unlock()
	dmst.ProtoBuf.Store(resp.ProtoId, entry.(*DummyProtocol))
}

//PLAYERS PROTOCOL
func (dp *DummyPlayer) Dispatch() {
	fmt.Printf("[+] Player %d started\n\n", dp.Id)
	for {
		//listen from Master
		buf := <-dp.FromMasterChan
		var msg DummyProtocolMsg
		json.Unmarshal(buf, &msg)
		switch msg.Type {
		case TYPES[0]: //PubKeySwitch --> no need for memory, just execute and send
			go dp.RunPubKeySwitch(msg)
		case TYPES[1]: //Refresh
			go dp.RunRefresh(msg)
		case TYPES[2]: //End
			go dp.End()
			break
		default:
			panic(errors.New("Unknown Protocol from Master"))
		}
	}
}

//Generates and send share to Master
func (dp *DummyPlayer) RunPubKeySwitch(msg DummyProtocolMsg) {
	fmt.Printf("[+] Player %d -- Received msg PCKS ID: %d from master\n\n", dp.Id, msg.Id)
	proto := dckks.NewPCKSProtocol(dp.Params, 3.2)
	var ct ring.Poly
	var pk rlwe.PublicKey
	var Ext DummyPCKSExt
	jsonString, _ := json.Marshal(msg.Extension)
	json.Unmarshal(jsonString, &Ext)
	err := ct.UnmarshalBinary(msg.Ct)
	utils.ThrowErr(err)
	err = pk.UnmarshalBinary(Ext.Pk)
	utils.ThrowErr(err)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(dp.sk, &pk, &ct, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := &DummyProtocolResp{Type: TYPES[0], Share: dat, PlayerId: dp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	fmt.Printf("[+] Player %d -- Sending Share PCKS ID: %d to master\n\n", dp.Id, msg.Id)
	fmt.Println("Len proto resp: ", len(dat))
	dp.ToMasterChan <- dat
}

func (dp *DummyPlayer) RunRefresh(msg DummyProtocolMsg) {
	var ct ring.Poly
	fmt.Printf("[+] Player %d -- Received msg Refresh ID: %d from master\n\n", dp.Id, msg.Id)
	ct.UnmarshalBinary(msg.Ct)
	var Ext DummyRefreshExt
	jsonString, _ := json.Marshal(msg.Extension)
	json.Unmarshal(jsonString, &Ext)
	crp, err := UnMarshalCrp(Ext.Crp)
	precision := Ext.Precision
	scale := Ext.Scale
	minLevel := Ext.MinLevel
	proto := dckks.NewRefreshProtocol(dp.Params, precision, 3.2)
	//proto := dp.BTP
	share := proto.AllocateShare(minLevel, dp.Params.MaxLevel())
	proto.GenShare(dp.sk, precision, dp.Params.LogSlots(), &ct, scale, crp, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := &DummyProtocolResp{Type: TYPES[1], Share: dat, PlayerId: dp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	fmt.Println("Len proto resp: ", len(dat))
	fmt.Printf("[+] Player %d -- Sending Share Refresh ID: %d to master\n\n", dp.Id, msg.Id)
	dp.ToMasterChan <- dat
}

func (dp *DummyPlayer) End() {
	close(dp.ToMasterChan)
}
