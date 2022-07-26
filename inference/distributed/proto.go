package distributed

import (
	"encoding/json"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/drlwe"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"sync"
)

var TYPES = []string{"PubKeySwitch", "Refresh", "End"}

type ProtocolType int16

const (
	CKSWITCH ProtocolType = iota
	REFRESH  ProtocolType = iota
	END      ProtocolType = iota
)

/*
	Dummy version of distributed protocols using channels for communication
*/
type Protocol struct {
	Protocol     interface{}      //instance of protocol
	Crp          drlwe.CKSCRP     //Common reference poly if any
	Ct           *ckks.Ciphertext //ciphertext of the protocol
	muxProto     sync.RWMutex
	Shares       []interface{}         //collects shares from parties
	Completion   int                   //counter to completion
	FeedbackChan chan *ckks.Ciphertext //final result of protocol
}

//Extension for PCKS
type PCKSExt struct {
	Pk []byte `json:"pk"` //Pub Key from Querier -> PubKeySwitch
}

//Extension for Refresh
type RefreshExt struct {
	Crp       []byte  `json:"crp"`       //Common poly from CRS -> Refresh
	Precision int     `json:"precision"` //Precision for instance of Refresh Protocol
	MinLevel  int     `json:"minlevel"`
	Scale     float64 `json:"scale"`
}

type ProtocolMsg struct {
	Type ProtocolType `json:"type"`
	Id   int          `json:"id"` //this is the id of the ct in the Enc Block Matrix, like i*row+j
	Ct   []byte       `json:"ct"` //ciphertext

	//Protocol Dependent
	Extension interface{} `json:"extension"`
}

type ProtocolResp struct {
	ProtoId  int          `json:"protoId"`
	Type     ProtocolType `json:"type"`
	PlayerId int          `json:"playerId"`
	Share    []byte       `json:"share"`
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
