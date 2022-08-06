package distributed

import (
	"encoding/json"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/drlwe"
	"sync"
)

var TYPES = []string{"PubKeySwitch", "Refresh", "End"}

type ProtocolType int16

const (
	CKSWITCH ProtocolType = iota
	REFRESH  ProtocolType = iota
	MASKING  ProtocolType = iota
	END      ProtocolType = iota
)

//Wrapper for distributed key switch or refresh
type Protocol struct {
	Protocol     interface{}      //instance of protocol
	Crp          drlwe.CKSCRP     //Common reference poly if any
	Ct           *ckks.Ciphertext //ciphertext of the protocol
	muxProto     sync.RWMutex
	Shares       []interface{}         //collects shares from parties
	Completion   int                   //counter to completion
	FeedbackChan chan *ckks.Ciphertext //final result of protocol
}

//Masking protocol for scenario data clear - model encrypted
type MaskProtocol struct {
	Ct           *ckks.Ciphertext //ct to be blindly decrypted
	Mask         *ckks.Plaintext  //used for masking
	Pt           *ckks.Plaintext  //result of decryption by server
	FeedbackChan chan *ckks.Plaintext
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

//Master to player
type ProtocolMsg struct {
	Type ProtocolType `json:"type"`
	Id   int          `json:"id"` //this is the id of the ct in the Enc Block Matrix, like i*row+j
	Ct   []byte       `json:"ct"` //ciphertext

	//Protocol Dependent
	Extension interface{} `json:"extension"`
}

//Used by players for replying to master
type ProtocolResp struct {
	ProtoId  int          `json:"protoId"`
	Type     ProtocolType `json:"type"`
	PlayerId int          `json:"playerId"`
	Share    []byte       `json:"share"`
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
