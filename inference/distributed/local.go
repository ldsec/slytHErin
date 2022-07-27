package distributed

import (
	"crypto/md5"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/dckks"
	"github.com/tuneinsight/lattigo/v3/drlwe"
	"github.com/tuneinsight/lattigo/v3/ring"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	lattigoUtils "github.com/tuneinsight/lattigo/v3/utils"
	"google.golang.org/grpc/benchmark/latency"
	"io"
	"net"
	"sync"
	"time"
)

/*
Local version of distributed protocols using  localhost for communication
*/

var DELIM = []byte{'\r', '\n', '\r', '\n'}
var TYP = uint8(255)
var KB = 1024
var MB = 1024 * KB

//var MAX_SIZE = 10 * MB //LogN = 14
var MAX_SIZE = 21 * MB //LogN = 15

var Network = &latency.Network{
	Kbps:    1024 * 1024, //1 Gbps
	Latency: 200 * time.Millisecond,
	MTU:     1500, // Ethernet
}

//HELPERS

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
	ProtoBuf *sync.Map //ct id -> protocol instance *Protocol
	sk       *rlwe.SecretKey
	Cpk      *rlwe.PublicKey
	Params   ckks.Parameters
	Parties  int
	//comms
	Addr        *net.TCPAddr
	PartiesAddr []*net.TCPAddr
	//for Ending
	runningMux    sync.RWMutex
	runningProtos int //counter for how many protos are running

	poolSize int //bound on parallel protocols
	Box      cipherUtils.CkksBox
	Done     chan bool //flag caller that master is done with all instances
}

type LocalPlayer struct {
	PCKS   *dckks.PCKSProtocol    //PubKeySwitch
	BTP    *dckks.RefreshProtocol //Bootstrap
	sk     *rlwe.SecretKey
	Cpk    *rlwe.PublicKey
	Params ckks.Parameters
	Id     int
	Addr   *net.TCPAddr
	Conn   net.Listener
}

func NewLocalMaster(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, parties int, partiesAddr []string, Box cipherUtils.CkksBox, poolSize int) (*LocalMaster, error) {
	master := new(LocalMaster)
	master.sk = sk
	master.Cpk = cpk
	master.Params = params
	master.Parties = parties
	//instance of protocol map
	master.ProtoBuf = new(sync.Map)

	master.PartiesAddr = make([]*net.TCPAddr, parties)
	master.Addr, _ = net.ResolveTCPAddr("tcp", partiesAddr[0])
	var err error
	for i := 1; i < parties; i++ {
		master.PartiesAddr[i], err = net.ResolveTCPAddr("tcp", partiesAddr[i])
		utils.ThrowErr(err)
	}
	master.poolSize = poolSize
	master.Box = Box
	return master, nil
}

func NewLocalPlayer(sk *rlwe.SecretKey, cpk *rlwe.PublicKey, params ckks.Parameters, id int, addr string) (*LocalPlayer, error) {
	player := new(LocalPlayer)
	player.sk = sk
	player.Cpk = cpk
	player.Params = params
	player.Id = id

	var err error
	player.Addr, err = net.ResolveTCPAddr("tcp", addr)
	utils.ThrowErr(err)
	listener, err := net.Listen("tcp", player.Addr.String())
	utils.ThrowErr(err)
	listener = Network.Listener(listener)
	player.Conn = listener.(net.Listener)
	return player, nil
}

//MASTER PROTOCOL

func (lmst *LocalMaster) spawnEvaluators(X *cipherUtils.EncInput, minLevel int, proto ProtocolType, pkQ *rlwe.PublicKey, ch chan []int) {
	var err error
	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		if proto == REFRESH && X.Blocks[i][j].Level() > minLevel {
			lmst.Box.Evaluator.ShallowCopy().DropLevel(X.Blocks[i][j], X.Blocks[i][j].Level()-minLevel)
		}
		X.Blocks[i][j], err = lmst.InitProto(proto, pkQ, X.Blocks[i][j], i*X.ColP+j)
		utils.ThrowErr(err)
	}
}

//starts protocol instances in parallel
func (lmst *LocalMaster) StartProto(proto ProtocolType, X *cipherUtils.EncInput, pkQ *rlwe.PublicKey, minLevel int) {
	var err error
	if proto == END {
		lmst.InitProto(proto, nil, nil, -1)
		return
	}
	if lmst.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				X.Blocks[i][j], err = lmst.InitProto(proto, pkQ, X.Blocks[i][j], i*X.ColP+j)
				utils.ThrowErr(err)
			}
		}
	} else if lmst.poolSize > 1 {
		//bounded threading

		ch := make(chan []int)
		var wg sync.WaitGroup
		//spawn consumers
		for i := 0; i < lmst.poolSize; i++ {
			wg.Add(1)
			go func() {
				lmst.spawnEvaluators(X, minLevel, proto, pkQ, ch)
				defer wg.Done()
			}()
		}
		//feed consumers
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				ch <- []int{i, j}
			}
		}
		close(ch)
		wg.Wait()
	}
}

//initiate protocol instance
func (lmst *LocalMaster) InitProto(proto ProtocolType, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) (*ckks.Ciphertext, error) {
	switch proto {
	case CKSWITCH:
		//prepare PubKeySwitch
		//fmt.Printf("[*] Master -- Registering PubKeySwitch ID: %d\n\n", ctId)
		protocol := dckks.NewPCKSProtocol(lmst.Params, 3.2)
		lmst.ProtoBuf.Store(ctId, &Protocol{
			Protocol:     protocol,
			Ct:           ct,
			Shares:       make([]interface{}, lmst.Parties),
			Completion:   0,
			FeedbackChan: make(chan *ckks.Ciphertext),
		})
		go lmst.RunPubKeySwitch(protocol.ShallowCopy(), pkQ, ct, ctId)

	case REFRESH:
		//prepare Refresh
		//fmt.Printf("[*] Master -- Registering Refresh ID: %d\n\n", ctId)
		var minLevel, logBound int
		var ok bool
		if minLevel, logBound, ok = dckks.GetMinimumLevelForBootstrapping(128, ct.Scale, lmst.Parties, lmst.Params.Q()); ok != true || minLevel+1 > lmst.Params.MaxLevel() {
			utils.ThrowErr(errors.New("Not enough levels to ensure correctness and 128 security"))
		}
		//creates proto instance
		protocol := dckks.NewRefreshProtocol(lmst.Params, logBound, 3.2)
		crs, _ := lattigoUtils.NewKeyedPRNG([]byte{'E', 'P', 'F', 'L'})
		crp := protocol.SampleCRP(lmst.Params.MaxLevel(), crs)
		lmst.ProtoBuf.Store(ctId, &Protocol{
			Protocol:     protocol,
			Ct:           ct,
			Crp:          crp,
			Shares:       make([]interface{}, lmst.Parties),
			Completion:   0,
			FeedbackChan: make(chan *ckks.Ciphertext),
		})
		go lmst.RunRefresh(protocol.ShallowCopy(), ct, crp, minLevel, logBound, ctId)
	case END:
		lmst.RunEnd()
		return nil, nil
	default:
		return nil, errors.New("Unknown protocol")
	}
	entry, ok := lmst.ProtoBuf.Load(ctId)
	if !ok {
		utils.ThrowErr(errors.New(fmt.Sprintf("Error fetching entry for id %d", ctId)))
	}
	res := <-entry.(*Protocol).FeedbackChan
	lmst.ProtoBuf.Delete(ctId)
	return res, nil
}

//reads reply from open connection to player
func (lmst *LocalMaster) Dispatch(c net.Conn) {
	buf, err := ReadFrom(c)
	utils.ThrowErr(err)
	//sum := md5.Sum(buf)
	//fmt.Printf("[*] Master received data %d B. Checksum: %x\n\n", len(buf), sum)
	var resp ProtocolResp
	err = json.Unmarshal(buf, &resp)
	utils.ThrowErr(err)
	switch resp.Type {
	case CKSWITCH:
		//key switch
		lmst.DispatchPCKS(resp)
	case REFRESH:
		lmst.DispatchRef(resp)
	default:
		utils.ThrowErr(errors.New("resp for unknown protocol"))
	}
}

//Runs the PCKS protocol from master and sends messages to players
func (lmst *LocalMaster) RunPubKeySwitch(proto *dckks.PCKSProtocol, pkQ *rlwe.PublicKey, ct *ckks.Ciphertext, ctId int) {
	//create protocol instance
	entry, ok := lmst.ProtoBuf.Load(ctId)
	if !ok {
		utils.ThrowErr(errors.New(fmt.Sprintf("Error fetching entry for id %d", ctId)))
	}
	entry.(*Protocol).muxProto.Lock()
	//proto := entry.(*Protocol).Proto.(*dckks.PCKSProtocol)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(lmst.sk, pkQ, ct.Value[1], share)
	entry.(*Protocol).Shares[0] = share
	entry.(*Protocol).Completion++
	entry.(*Protocol).muxProto.Unlock()
	lmst.ProtoBuf.Store(ctId, entry.(*Protocol))

	dat, err := ct.Value[1].MarshalBinary()
	utils.ThrowErr(err)
	dat2, err := pkQ.MarshalBinary()
	utils.ThrowErr(err)
	msg := ProtocolMsg{Type: CKSWITCH, Id: ctId, Ct: dat}
	msg.Extension = PCKSExt{Pk: dat2}
	buf, err := json.Marshal(msg)
	sum := md5.Sum(buf)
	utils.ThrowErr(err)

	// send message and listen for reply:
	// every proto instance uses its proper socket in order to have a dedicated channel for
	// its message. This might not be efficient, but we avoid define a logic for reassembling packets
	// belonging to the same instance at application level
	for i := 1; i < lmst.Parties; i++ {
		fmt.Printf("[*] Master -- Sending key swith init %d B ID: %d to %d. Checksum: %x\n\n", len(buf), msg.Id, i, sum)
		addr := lmst.PartiesAddr[i]
		c, err := net.Dial("tcp", addr.String())
		utils.ThrowErr(err)
		c, err = Network.Conn(c)
		utils.ThrowErr(err)
		go func(c net.Conn, buf []byte) {
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
	entry, ok := lmst.ProtoBuf.Load(ctId)
	if !ok {
		utils.ThrowErr(errors.New(fmt.Sprintf("Error fetching entry for id %d", ctId)))
	}
	share := proto.AllocateShare(minLevel, lmst.Params.MaxLevel())
	proto.GenShare(lmst.sk, logBound, lmst.Params.LogSlots(), ct.Value[1], ct.Scale, crp, share)
	entry.(*Protocol).Shares[0] = share
	entry.(*Protocol).Completion++
	lmst.ProtoBuf.Store(ctId, entry.(*Protocol))
	dat, err := ct.Value[1].MarshalBinary()
	utils.ThrowErr(err)
	var dat2 []byte

	if dat2, err = MarshalCrp(crp); err != nil {
		utils.ThrowErr(err)
	}
	msg := ProtocolMsg{Type: REFRESH, Id: ctId, Ct: dat}
	msg.Extension = RefreshExt{
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
		addr := lmst.PartiesAddr[i]
		c, err := net.Dial("tcp", addr.String())
		utils.ThrowErr(err)
		c, err = Network.Conn(c)
		utils.ThrowErr(err)
		go func(c net.Conn, buf []byte) {
			defer c.Close()
			err = WriteTo(c, buf)
			utils.ThrowErr(err)
			lmst.Dispatch(c)
		}(c, buf)
	}
}

func (lmst *LocalMaster) RunEnd() {
	msg := ProtocolMsg{Type: END, Id: 0, Ct: nil}
	msg.Extension = struct{}{}
	buf, err := json.Marshal(msg)
	utils.ThrowErr(err)

	// send message and close
	for i := 1; i < lmst.Parties; i++ {
		//fmt.Printf("[*] Master -- Sending end to %d. \n\n", i)
		addr := lmst.PartiesAddr[i]
		c, err := net.DialTCP("tcp", nil, addr)
		utils.ThrowErr(err)
		go func(c *net.TCPConn, buf []byte) {
			defer c.Close()
			err = WriteTo(c, buf)
			utils.ThrowErr(err)
		}(c, buf)
	}
}

//Listen for shares and aggregates
func (lmst *LocalMaster) DispatchPCKS(resp ProtocolResp) {
	if resp.PlayerId == 0 {
		//ignore invalid
		return
	}
	//fmt.Printf("[*] Master -- Received share of PCKS ID: %d from %d\n\n", resp.ProtoId, resp.PlayerId)
	entry, ok := lmst.ProtoBuf.Load(resp.ProtoId)
	if !ok {
		return
	}
	entry.(*Protocol).muxProto.Lock()
	proto := entry.(*Protocol).Protocol.(*dckks.PCKSProtocol)
	ct := entry.(*Protocol).Ct

	share := new(drlwe.PCKSShare)
	err := share.UnmarshalBinary(resp.Share)
	utils.ThrowErr(err)
	entry.(*Protocol).Shares[resp.PlayerId] = share
	entry.(*Protocol).Completion++
	if entry.(*Protocol).Completion == lmst.Parties {
		//finalize
		for i := 1; i < lmst.Parties; i++ {
			proto.AggregateShare(
				entry.(*Protocol).Shares[i].(*drlwe.PCKSShare),
				entry.(*Protocol).Shares[0].(*drlwe.PCKSShare),
				entry.(*Protocol).Shares[0].(*drlwe.PCKSShare))
		}
		ctSw := ckks.NewCiphertext(lmst.Params, 1, ct.Level(), ct.Scale)
		proto.KeySwitch(ct, entry.(*Protocol).Shares[0].(*drlwe.PCKSShare), ctSw)
		entry.(*Protocol).FeedbackChan <- ctSw
	}
	lmst.ProtoBuf.Store(resp.ProtoId, entry.(*Protocol))
	entry.(*Protocol).muxProto.Unlock()
}

//Listen for shares and Finalize
func (lmst *LocalMaster) DispatchRef(resp ProtocolResp) {
	if resp.PlayerId == 0 {
		//ignore invalid
		return
	}
	//fmt.Printf("[*] Master -- Received share of Refresh ID: %d from %d\n\n", resp.ProtoId, resp.PlayerId)
	entry, ok := lmst.ProtoBuf.Load(resp.ProtoId)
	if !ok {
		return
	}
	entry.(*Protocol).muxProto.Lock()
	proto := entry.(*Protocol).Protocol.(*dckks.RefreshProtocol)
	crp := entry.(*Protocol).Crp
	ct := entry.(*Protocol).Ct

	share := new(dckks.RefreshShare)
	err := share.UnmarshalBinary(resp.Share)
	utils.ThrowErr(err)
	entry.(*Protocol).Shares[resp.PlayerId] = share
	entry.(*Protocol).Completion++
	if entry.(*Protocol).Completion == lmst.Parties {
		//finalize
		for i := 1; i < lmst.Parties; i++ {
			proto.AggregateShare(
				entry.(*Protocol).Shares[i].(*dckks.RefreshShare),
				entry.(*Protocol).Shares[0].(*dckks.RefreshShare),
				entry.(*Protocol).Shares[0].(*dckks.RefreshShare))
		}
		ctFresh := ckks.NewCiphertext(lmst.Params, 1, lmst.Params.MaxLevel(), ct.Scale)
		proto.Finalize(ct, lmst.Params.LogSlots(), crp, entry.(*Protocol).Shares[0].(*dckks.RefreshShare), ctFresh)
		entry.(*Protocol).FeedbackChan <- ctFresh
	}

	lmst.ProtoBuf.Store(resp.ProtoId, entry.(*Protocol))
	entry.(*Protocol).muxProto.Unlock()
}

//PLAYERS PROTOCOL

//Accepts an incoming TCP connection and handles it (blocking)
func (lp *LocalPlayer) Listen() {
	//fmt.Printf("[+] Player %d started at %s\n\n", lp.Id, lp.Addr.String())
	for {
		c, err := lp.Conn.Accept()
		if err != nil {
			//player is ending
			return
		}
		//fmt.Printf("[+] Player %d accepted connection\n\n", lp.Id)
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

		//sum := md5.Sum(netData)
		//fmt.Printf("[+] Player %d received data %d B. Checksum: %x\n\n", lp.Id, len(netData), sum)
		var msg ProtocolMsg
		err = json.Unmarshal(netData, &msg)
		utils.ThrowErr(err)
		switch msg.Type {
		case CKSWITCH: //PubKeySwitch
			lp.RunPubKeySwitch(c, msg)
		case REFRESH: //Refresh
			lp.RunRefresh(c, msg)
		case END: //End
			lp.End(c)
			return
		default:
			panic(errors.New("Unknown Protocol from Master"))
		}
	}
}

//Generates and send share to Master
func (lp *LocalPlayer) RunPubKeySwitch(c net.Conn, msg ProtocolMsg) {
	//fmt.Printf("[+] Player %d -- Received msg PCKS ID: %d from master\n\n", lp.Id, msg.Id)
	proto := dckks.NewPCKSProtocol(lp.Params, 3.2)
	var ct ring.Poly
	var pk rlwe.PublicKey
	var Ext PCKSExt
	jsonString, err := json.Marshal(msg.Extension)
	utils.ThrowErr(err)
	err = json.Unmarshal(jsonString, &Ext)
	utils.ThrowErr(err)
	err = ct.UnmarshalBinary(msg.Ct)
	utils.ThrowErr(err)
	err = pk.UnmarshalBinary(Ext.Pk)
	utils.ThrowErr(err)
	share := proto.AllocateShare(ct.Level())
	proto.GenShare(lp.sk, &pk, &ct, share)
	dat, err := share.MarshalBinary()
	utils.ThrowErr(err)
	resp := ProtocolResp{Type: CKSWITCH, Share: dat, PlayerId: lp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	utils.ThrowErr(err)
	sum := md5.Sum(dat)
	fmt.Printf("[+] Player %d -- Sending Share (%d B) PCKS ID: %d to master. Checksum: %x \n\n", lp.Id, len(dat), msg.Id, sum)
	err = WriteTo(c, dat)
	utils.ThrowErr(err)
}

func (lp *LocalPlayer) RunRefresh(c net.Conn, msg ProtocolMsg) {
	var ct ring.Poly
	//fmt.Printf("[+] Player %d -- Received msg Refresh ID: %d from master\n\n", lp.Id, msg.Id)
	err := ct.UnmarshalBinary(msg.Ct)
	utils.ThrowErr(err)
	var Ext RefreshExt
	jsonString, err := json.Marshal(msg.Extension)
	utils.ThrowErr(err)
	err = json.Unmarshal(jsonString, &Ext)
	utils.ThrowErr(err)
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
	resp := ProtocolResp{Type: REFRESH, Share: dat, PlayerId: lp.Id, ProtoId: msg.Id}
	dat, err = json.Marshal(resp)
	utils.ThrowErr(err)
	sum := md5.Sum(dat)
	fmt.Printf("[+] Player %d -- Sending Share (%d B) Refresh ID: %d to master. Checksum: %x \n\n", lp.Id, len(dat), msg.Id, sum)
	err = WriteTo(c, dat)
	utils.ThrowErr(err)
	utils.ThrowErr(err)
}

func (lp *LocalPlayer) End(c net.Conn) {
	//fmt.Printf("[+] Player %d terminating!\n\n", lp.Id)
	c.Close()
	lp.Conn.Close()
	return
}
