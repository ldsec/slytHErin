package distributed

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
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
	"io"
	"net"
	"sync"
)

/*
Local version of distributed protocols using  localhost for communication
*/

var DELIM = []byte{'\r', '\n', '\r', '\n'}
var TYP = uint8(255)
var KB = 1024
var MB = 1024 * KB
var MAX_SIZE = 10 * MB

//HELPERS

//custom Split function for bufio.Scanner to read until the DELIM sequence
func Split(data []byte, atEOF bool) (advance int, token []byte, err error) {
	//fmt.Println("Scanner: len of data", len(data))
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	if i := bytes.Index(data, DELIM); i >= 0 {
		return i + len(DELIM), data[0:i], nil
	}
	if atEOF {
		return len(data), data, nil
	}
	return 0, nil, nil
}

//Append DELIM sequence to bytes
func DelimEncode(buf []byte) []byte {
	return append(buf, DELIM...)
}

//write TLV value
func WriteTo(c io.Writer, buf []byte) error {
	err := binary.Write(c, binary.LittleEndian, TYP) //1-byte type
	if err != nil {
		return err
	}
	err = binary.Write(c, binary.LittleEndian, uint32(len(buf))) //4-byte len
	if err != nil {
		return err
	}
	err = binary.Write(c, binary.LittleEndian, buf)
	return err
}

//reads TLV value
func ReadFrom(c io.Reader) ([]byte, error) {
	var typ uint8
	err := binary.Read(c, binary.LittleEndian, &typ)
	if err != nil {
		return nil, err
	}
	if typ != TYP {
		return nil, errors.New("Not TYP")
	}
	var l uint32
	err = binary.Read(c, binary.LittleEndian, &l)
	if err != nil {
		return nil, err
	}
	if int(l) > MAX_SIZE {
		return nil, errors.New("Payload too large")
	}
	buf := make([]byte, l)
	err = binary.Read(c, binary.LittleEndian, buf)
	return buf, err
}

type LocalMaster struct {
	ProtoBuf *sync.Map //ct id -> protocol instance *DummyProtocol
	sk       *rlwe.SecretKey
	Cpk      *rlwe.PublicKey
	Params   ckks.Parameters
	Parties  int
	//comms
	Addr        *net.TCPAddr
	PartiesAddr []*net.TCPAddr
	//for Ending
	runningMux    sync.RWMutex
	runningProtos int       //counter for how many protos are running
	Done          chan bool //flag caller that master is done with all instances
}

type LocalPlayer struct {
	PCKS   *dckks.PCKSProtocol    //PubKeySwitch
	BTP    *dckks.RefreshProtocol //Bootstrap
	sk     *rlwe.SecretKey
	Cpk    *rlwe.PublicKey
	Params ckks.Parameters
	Id     int
	Addr   *net.TCPAddr
	Conn   *net.TCPListener
}

func NewLocalMaster(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, parties int, partiesAddr []string) (*LocalMaster, error) {
	master := new(LocalMaster)
	master.sk = sk
	master.Cpk = cpk
	master.Params = params
	master.Parties = parties
	//instance of protocol map
	master.ProtoBuf = new(sync.Map)

	master.PartiesAddr = make([]*net.TCPAddr, parties)
	master.Addr, _ = net.ResolveTCPAddr("tcp", partiesAddr[0])
	for i := 1; i < parties; i++ {
		master.PartiesAddr[i], _ = net.ResolveTCPAddr("tcp", partiesAddr[i])
	}
	return master, nil
}

func NewLocalPlayer(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, id int, addr string) (*LocalPlayer, error) {
	player := new(LocalPlayer)
	player.sk = sk
	player.Cpk = cpk
	player.Params = params
	player.Id = id

	player.Addr, _ = net.ResolveTCPAddr("tcp", addr)
	player.Conn, _ = net.ListenTCP("tcp", player.Addr)
	return player, nil
}

//MASTER PROTOCOL

//starts a new protocol instance and runs it
func (lmst *LocalMaster) InitProto(proto string, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) (*ckks.Ciphertext, error) {
	switch proto {
	case TYPES[0]:
		//prepare PubKeySwitch
		fmt.Printf("[*] Master -- Registering PubKeySwitch ID: %d\n\n", ctId)
		protocol := dckks.NewPCKSProtocol(lmst.Params, 3.2)
		lmst.ProtoBuf.Store(ctId, &DummyProtocol{
			Protocol:     protocol,
			Ct:           ct,
			Shares:       make([]interface{}, lmst.Parties),
			Completion:   0,
			FeedbackChan: make(chan *ckks.Ciphertext),
		})
		go lmst.RunPubKeySwitch(protocol.ShallowCopy(), pkQ, ct, ctId)

	case TYPES[1]:
		//prepare Refresh
		fmt.Printf("[*] Master -- Registering Refresh ID: %d\n\n", ctId)
		var minLevel, logBound int
		var ok bool
		if minLevel, logBound, ok = dckks.GetMinimumLevelForBootstrapping(128, ct.Scale, lmst.Parties, lmst.Params.Q()); ok != true || minLevel+1 > lmst.Params.MaxLevel() {
			utils.ThrowErr(errors.New("Not enough levels to ensure correctness and 128 security"))
		}
		//creates proto instance
		protocol := dckks.NewRefreshProtocol(lmst.Params, logBound, 3.2)
		crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})
		crp := protocol.SampleCRP(lmst.Params.MaxLevel(), crs)
		lmst.ProtoBuf.Store(ctId, &DummyProtocol{
			Protocol:     protocol,
			Ct:           ct,
			Crp:          crp,
			Shares:       make([]interface{}, lmst.Parties),
			Completion:   0,
			FeedbackChan: make(chan *ckks.Ciphertext),
		})
		go lmst.RunRefresh(protocol.ShallowCopy(), ct, crp, minLevel, logBound, ctId)
	default:
		return nil, errors.New("Unknown protocol")
	}
	entry, _ := lmst.ProtoBuf.Load(ctId)
	res := <-entry.(*DummyProtocol).FeedbackChan
	lmst.ProtoBuf.Delete(ctId)
	return res, nil
}

//reads reply from open connection to player
func (lmst *LocalMaster) Dispatch(c *net.TCPConn) {
	buf, err := ReadFrom(c)
	utils.ThrowErr(err)
	sum := md5.Sum(buf)
	fmt.Printf("[*] Master received data %d B. Checksum: %x\n\n", len(buf), sum)
	var resp DummyProtocolResp
	err = json.Unmarshal(buf, &resp)
	utils.ThrowErr(err)
	switch resp.Type {
	case TYPES[0]:
		//key switch
		go lmst.DispatchPCKS(resp)
	case TYPES[1]:
		go lmst.DispatchRef(resp)
	default:
		utils.ThrowErr(errors.New("resp for unknown protocol"))
	}
}

//Runs the PCKS protocol from master and sends messages to players
func (lmst *LocalMaster) RunPubKeySwitch(proto *dckks.PCKSProtocol, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) {
	//create protocol instance
	entry, _ := lmst.ProtoBuf.Load(ctId)
	entry.(*DummyProtocol).muxProto.Lock()
	//proto := entry.(*DummyProtocol).Proto.(*dckks.PCKSProtocol)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(lmst.sk, pkQ, ct.Value[1], share)
	entry.(*DummyProtocol).Shares[0] = share
	entry.(*DummyProtocol).Completion++
	entry.(*DummyProtocol).muxProto.Unlock()
	lmst.ProtoBuf.Store(ctId, entry.(*DummyProtocol))

	dat, _ := ct.Value[1].MarshalBinary()
	dat2, _ := pkQ.MarshalBinary()
	msg := DummyProtocolMsg{Type: TYPES[0], Id: ctId, Ct: dat}
	msg.Extension = DummyPCKSExt{Pk: dat2}
	buf, err := json.Marshal(msg)
	sum := md5.Sum(buf)
	utils.ThrowErr(err)

	// send message and listen for reply:
	// every proto instance uses its proper socket in order to have a dedicated channel for
	// its message. This might not be efficient, but we avoid define a logic for reassembling packets
	// belonging to the same instance at application level
	for i := 1; i < lmst.Parties; i++ {
		fmt.Printf("[*] Master -- Sending key swith init %d B ID: %d to %d. Checksum: %x\n\n", len(buf), msg.Id, i, sum)
		//c := lmst.PartiesConn[i]
		addr := lmst.PartiesAddr[i]
		c, err := net.DialTCP("tcp", nil, addr)
		utils.ThrowErr(err)
		go func(c *net.TCPConn, buf []byte) {
			defer c.Close()
			err = WriteTo(c, buf)
			utils.ThrowErr(err)
			lmst.Dispatch(c)
		}(c, buf)
	}
}

//Runs the Refresh protocol from master and sends messages to players
func (lmst *LocalMaster) RunRefresh(proto *dckks.RefreshProtocol, ct *ckks.Ciphertext, crp drlwe.CKSCRP, minLevel int, logBound int, ctId int) {
	//setup
	entry, _ := lmst.ProtoBuf.Load(ctId)
	share := proto.AllocateShare(minLevel, lmst.Params.MaxLevel())
	proto.GenShare(lmst.sk, logBound, lmst.Params.LogSlots(), ct.Value[1], ct.Scale, crp, share)
	entry.(*DummyProtocol).Shares[0] = share
	entry.(*DummyProtocol).Completion++
	lmst.ProtoBuf.Store(ctId, entry.(*DummyProtocol))
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
	sum := md5.Sum(buf)
	utils.ThrowErr(err)

	// send message and listen for reply:
	// every proto instance uses its proper socket in order to have a dedicated channel for
	// its message. This might not be efficient, but we avoid define a logic for reassembling packets
	// belonging to the same instance at application level
	for i := 1; i < lmst.Parties; i++ {
		fmt.Printf("[*] Master -- Sending refresh init (len %d) ID: %d to %d. Checksum: %x\n\n", len(buf), msg.Id, i, sum)
		//c := lmst.PartiesConn[i]
		addr := lmst.PartiesAddr[i]
		c, err := net.DialTCP("tcp", nil, addr)
		utils.ThrowErr(err)
		go func(c *net.TCPConn, buf []byte) {
			defer c.Close()
			err = WriteTo(c, buf)
			utils.ThrowErr(err)
			lmst.Dispatch(c)
		}(c, buf)
	}
}

//Listen for shares and aggregates
func (lmst *LocalMaster) DispatchPCKS(resp DummyProtocolResp) {
	if resp.PlayerId == 0 {
		//ignore invalid
		return
	}
	fmt.Printf("[*] Master -- Received share of PCKS ID: %d from %d\n\n", resp.ProtoId, resp.PlayerId)
	entry, ok := lmst.ProtoBuf.Load(resp.ProtoId)
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
	if entry.(*DummyProtocol).Completion == lmst.Parties {
		//finalize
		for i := 1; i < lmst.Parties; i++ {
			proto.AggregateShare(
				entry.(*DummyProtocol).Shares[i].(*drlwe.PCKSShare),
				entry.(*DummyProtocol).Shares[0].(*drlwe.PCKSShare),
				entry.(*DummyProtocol).Shares[0].(*drlwe.PCKSShare))
		}
		ctSw := ckks.NewCiphertext(lmst.Params, 1, ct.Level(), ct.Scale)
		proto.KeySwitch(ct, entry.(*DummyProtocol).Shares[0].(*drlwe.PCKSShare), ctSw)
		entry.(*DummyProtocol).FeedbackChan <- ctSw
	}
	entry.(*DummyProtocol).muxProto.Unlock()
	lmst.ProtoBuf.Store(resp.ProtoId, entry.(*DummyProtocol))
}

//Listen for shares and Finalize
func (lmst *LocalMaster) DispatchRef(resp DummyProtocolResp) {
	if resp.PlayerId == 0 {
		//ignore invalid
		return
	}
	fmt.Printf("[*] Master -- Received share of Refresh ID: %d from %d\n\n", resp.ProtoId, resp.PlayerId)
	entry, ok := lmst.ProtoBuf.Load(resp.ProtoId)
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
	if entry.(*DummyProtocol).Completion == lmst.Parties {
		//finalize
		for i := 1; i < lmst.Parties; i++ {
			proto.AggregateShare(
				entry.(*DummyProtocol).Shares[i].(*dckks.RefreshShare),
				entry.(*DummyProtocol).Shares[0].(*dckks.RefreshShare),
				entry.(*DummyProtocol).Shares[0].(*dckks.RefreshShare))
		}
		ctFresh := ckks.NewCiphertext(lmst.Params, 1, lmst.Params.MaxLevel(), ct.Scale)
		proto.Finalize(ct, lmst.Params.LogSlots(), crp, entry.(*DummyProtocol).Shares[0].(*dckks.RefreshShare), ctFresh)
		entry.(*DummyProtocol).FeedbackChan <- ctFresh
	}
	entry.(*DummyProtocol).muxProto.Unlock()
	lmst.ProtoBuf.Store(resp.ProtoId, entry.(*DummyProtocol))
}

//PLAYERS PROTOCOL

//Accepts an incoming TCP connection and handles it (blocking)
func (lp *LocalPlayer) Listen() {
	fmt.Printf("[+] Player %d started at %s\n\n", lp.Id, lp.Addr.String())
	for {
		c, err := lp.Conn.Accept()
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("[+] Player %d accepted connection\n\n", lp.Id)
		go lp.Dispatch(c)
	}
}

//Handler for the connection
func (lp *LocalPlayer) Dispatch(c net.Conn) {
	//listen from Master
	for {
		netData, err := ReadFrom(c)
		if err == io.EOF {
			c.Close()
			break
		}
		utils.ThrowErr(err)
		sum := md5.Sum(netData)
		fmt.Printf("[+] Player %d received data %d B. Checksum: %x\n\n", lp.Id, len(netData), sum)
		var msg DummyProtocolMsg
		json.Unmarshal(netData, &msg)
		switch msg.Type {
		case TYPES[0]: //PubKeySwitch
			lp.RunPubKeySwitch(c, msg)
		case TYPES[1]: //Refresh
			lp.RunRefresh(c, msg)
		case TYPES[2]: //End
			lp.End()
			break
		default:
			panic(errors.New("Unknown Protocol from Master"))
		}
	}
}

//Generates and send share to Master
func (lp *LocalPlayer) RunPubKeySwitch(c net.Conn, msg DummyProtocolMsg) {
	fmt.Printf("[+] Player %d -- Received msg PCKS ID: %d from master\n\n", lp.Id, msg.Id)
	proto := dckks.NewPCKSProtocol(lp.Params, 3.2)
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
	proto.GenShare(lp.sk, &pk, &ct, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := DummyProtocolResp{Type: TYPES[0], Share: dat, PlayerId: lp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	utils.ThrowErr(err)
	sum := md5.Sum(dat)
	fmt.Printf("[+] Player %d -- Sending Share (%d B) PCKS ID: %d to master. Checksum: %x \n\n", lp.Id, len(dat), msg.Id, sum)
	err = WriteTo(c, dat)
	utils.ThrowErr(err)
}

func (lp *LocalPlayer) RunRefresh(c net.Conn, msg DummyProtocolMsg) {
	var ct ring.Poly
	fmt.Printf("[+] Player %d -- Received msg Refresh ID: %d from master\n\n", lp.Id, msg.Id)
	ct.UnmarshalBinary(msg.Ct)
	var Ext DummyRefreshExt
	jsonString, _ := json.Marshal(msg.Extension)
	json.Unmarshal(jsonString, &Ext)
	crp, err := UnMarshalCrp(Ext.Crp)
	precision := Ext.Precision
	scale := Ext.Scale
	minLevel := Ext.MinLevel
	proto := dckks.NewRefreshProtocol(lp.Params, precision, 3.2)
	//proto := dp.BTP
	share := proto.AllocateShare(minLevel, lp.Params.MaxLevel())
	proto.GenShare(lp.sk, precision, lp.Params.LogSlots(), &ct, scale, crp, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := DummyProtocolResp{Type: TYPES[1], Share: dat, PlayerId: lp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	utils.ThrowErr(err)
	sum := md5.Sum(dat)
	fmt.Printf("[+] Player %d -- Sending Share (%d B) Refresh ID: %d to master. Checksum: %x \n\n", lp.Id, len(dat), msg.Id, sum)
	err = WriteTo(c, dat)
	utils.ThrowErr(err)
	utils.ThrowErr(err)
}

func (lp *LocalPlayer) End() {

}
