package distributed

import (
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"io"
	"net"
	"sync"
)

type Client struct {
	ProtoBuf *sync.Map //ct id -> protocol instance *Protocol

	//comms
	ServerAddr *net.TCPAddr

	//for Ending
	runningMux    sync.RWMutex
	runningProtos int //counter for how many protos are running

	poolSize int //bound on parallel protocols
	Box      cipherUtils.CkksBox
	done     chan bool //flag caller that client is done with all instances
}

type Server struct {
	Box  cipherUtils.CkksBox
	Addr *net.TCPAddr
	Conn net.Listener
}

func NewClient(ServerAddr string, Box cipherUtils.CkksBox, poolSize int) (*Client, error) {
	cl := new(Client)
	cl.Box = Box
	cl.poolSize = poolSize
	//instance of protocol map
	cl.ProtoBuf = new(sync.Map)
	var err error
	cl.ServerAddr, err = net.ResolveTCPAddr("tcp", ServerAddr)

	return cl, err
}

func NewServer(Box cipherUtils.CkksBox, addr string) (*Server, error) {
	server := new(Server)

	var err error
	server.Addr, err = net.ResolveTCPAddr("tcp", addr)
	listener, err := net.Listen("tcp", server.Addr.String())
	listener = Network.Listener(listener)
	server.Conn = listener.(net.Listener)
	server.Box = Box
	go server.listen()
	return server, err
}

//MASTER PROTOCOL

func (Cl *Client) spawnEvaluators(X *cipherUtils.EncInput, res *cipherUtils.PlainInput, proto ProtocolType, ch chan []int) {
	var err error
	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		res.Blocks[i][j], err = Cl.initProto(proto, X.Blocks[i][j], i*X.ColP+j)
		utils.ThrowErr(err)
	}
}

//starts protocol instances in parallel
func (Cl *Client) StartProto(proto ProtocolType, X *cipherUtils.EncInput) *cipherUtils.PlainInput {
	var err error
	if proto == END {
		Cl.initProto(proto, nil, -1)
		return nil
	}
	res := &cipherUtils.PlainInput{
		Blocks:    make([][]*ckks.Plaintext, X.RowP),
		RowP:      X.RowP,
		ColP:      X.ColP,
		InnerRows: X.InnerRows,
		InnerCols: X.InnerCols,
	}
	for i := range res.Blocks {
		res.Blocks[i] = make([]*ckks.Plaintext, res.ColP)
	}
	if Cl.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				res.Blocks[i][j], err = Cl.initProto(proto, X.Blocks[i][j], i*X.ColP+j)
				utils.ThrowErr(err)
			}
		}
	} else if Cl.poolSize > 1 {
		//bounded threading

		ch := make(chan []int)
		var wg sync.WaitGroup
		//spawn consumers
		for i := 0; i < Cl.poolSize; i++ {
			wg.Add(1)
			go func() {
				Cl.spawnEvaluators(X, res, proto, ch)
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
	return res
}

//initiate protocol instance
func (Cl *Client) initProto(proto ProtocolType, ct *ckks.Ciphertext, ctId int) (*ckks.Plaintext, error) {
	switch proto {
	case MASKING:
		Cl.ProtoBuf.Store(ctId, &MaskProtocol{
			Ct:           ct,
			Pt:           ckks.NewPlaintext(Cl.Box.Params, ct.Level(), ct.Scale),
			FeedbackChan: make(chan *ckks.Plaintext),
		})
		go Cl.RunMasking(ctId)

	case END:
		Cl.RunEnd()
		return nil, nil
	default:
		return nil, errors.New("Unknown protocol")
	}
	entry, ok := Cl.ProtoBuf.Load(ctId)
	if !ok {
		utils.ThrowErr(errors.New(fmt.Sprintf("Error fetching entry for id %d", ctId)))
	}
	res := <-entry.(*MaskProtocol).FeedbackChan
	Cl.ProtoBuf.Delete(ctId)
	return res, nil
}

//reads reply from open connection to player
func (Cl *Client) Dispatch(c net.Conn) {
	buf, err := ReadFrom(c)
	utils.ThrowErr(err)
	//sum := md5.Sum(buf)
	//fmt.Printf("[*] Master received data %d B. Checksum: %x\n\n", len(buf), sum)
	var resp ProtocolMsg
	err = json.Unmarshal(buf, &resp)
	utils.ThrowErr(err)
	switch resp.Type {
	case MASKING:
		Cl.DispatchMasking(resp)
	default:
		utils.ThrowErr(errors.New("resp for unknown protocol"))
	}
}

//Runs the Masking protocol
func (Cl *Client) RunMasking(ctId int) {
	//create protocol instance
	entry, ok := Cl.ProtoBuf.Load(ctId)
	if !ok {
		utils.ThrowErr(errors.New(fmt.Sprintf("Error fetching entry for id %d", ctId)))
	}
	ct := entry.(*MaskProtocol).Ct
	mask := cipherUtils.Mask(ct, cipherUtils.BoxShallowCopy(Cl.Box))
	//update entry
	entry.(*MaskProtocol).Mask = mask
	entry.(*MaskProtocol).Ct = ct
	Cl.ProtoBuf.Store(ctId, entry)

	dat, err := ct.MarshalBinary()
	utils.ThrowErr(err)
	msg := ProtocolMsg{Type: MASKING, Id: ctId, Ct: dat}
	buf, err := json.Marshal(msg)
	sum := md5.Sum(buf)
	utils.ThrowErr(err)

	// send message and listen for reply:

	fmt.Printf("[*] Client -- Sending masking init %d B ID: %d to server. Checksum: %x\n\n", len(buf), msg.Id, sum)
	addr := Cl.ServerAddr
	c, err := net.Dial("tcp", addr.String())
	utils.ThrowErr(err)
	c, err = Network.Conn(c)
	utils.ThrowErr(err)
	go func(c net.Conn, buf []byte) {
		defer c.Close()
		err = WriteTo(c, buf)
		utils.ThrowErr(err)
		Cl.Dispatch(c)
	}(c, buf)
}

func (Cl *Client) RunEnd() {
	msg := ProtocolMsg{Type: END, Id: 0, Ct: nil}
	buf, err := json.Marshal(msg)
	utils.ThrowErr(err)

	// send message and close

	addr := Cl.ServerAddr
	c, err := net.DialTCP("tcp", nil, addr)
	utils.ThrowErr(err)
	go func(c *net.TCPConn, buf []byte) {
		defer c.Close()
		err = WriteTo(c, buf)
		utils.ThrowErr(err)
	}(c, buf)

}

//Listen for shares and aggregates
func (Cl *Client) DispatchMasking(resp ProtocolMsg) {
	entry, ok := Cl.ProtoBuf.Load(resp.Id)
	if !ok {
		return
	}
	pt := entry.(*MaskProtocol).Pt
	utils.ThrowErr(pt.Value.UnmarshalBinary(resp.Ct))
	cipherUtils.UnMask(pt, entry.(*MaskProtocol).Mask, cipherUtils.BoxShallowCopy(Cl.Box))
	Cl.ProtoBuf.Store(resp.Id, entry.(*MaskProtocol))
	entry.(*MaskProtocol).FeedbackChan <- pt
}

//Server PROTOCOL

//Accepts an incoming TCP connection and handles it (blocking)
func (s *Server) listen() {
	//fmt.Printf("[+] Player %d started at %s\n\n", lp.Id, lp.Addr.String())
	for {
		c, err := s.Conn.Accept()
		if err != nil {
			//player is ending
			return
		}
		//fmt.Printf("[+] Player %d accepted connection\n\n", lp.Id)
		go s.Dispatch(c)
	}
}

//Handler for the connection
func (s *Server) Dispatch(c net.Conn) {
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
		case MASKING:
			s.RunMask(c, msg)
		case END: //End
			s.End(c)
			return
		default:
			panic(errors.New("Unknown Protocol from Master"))
		}
	}
}

//Generates and send share to Master
func (s *Server) RunMask(c net.Conn, msg ProtocolMsg) {
	//fmt.Printf("[+] Player %d -- Received msg PCKS ID: %d from master\n\n", lp.Id, msg.Id)
	ct := new(ckks.Ciphertext)
	ct.UnmarshalBinary(msg.Ct)
	box := cipherUtils.BoxShallowCopy(s.Box)
	pt := box.Decryptor.DecryptNew(ct)
	dat, err := pt.Value.MarshalBinary()
	utils.ThrowErr(err)
	resp := ProtocolMsg{Type: MASKING, Id: msg.Id, Ct: dat}
	dat, err = json.Marshal(resp)
	utils.ThrowErr(err)
	sum := md5.Sum(dat)
	fmt.Printf("[+] Server -- Sending Masked Dec %d to client %d B. Checksum: %x \n\n", msg.Id, len(dat), sum)
	err = WriteTo(c, dat)
	utils.ThrowErr(err)
}

func (s *Server) End(c net.Conn) {
	//fmt.Printf("[+] Player %d terminating!\n\n", lp.Id)
	c.Close()
	s.Conn.Close()
	return
}
