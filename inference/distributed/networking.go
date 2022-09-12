package distributed

//Contains networking technicalities and constants

import (
	"encoding/binary"
	"errors"
	"google.golang.org/grpc/benchmark/latency"
	"io"
	"time"
)

var DELIM = []byte{'\r', '\n', '\r', '\n'}
var TYP = uint8(255)
var KB = 1024
var MB = 1024 * KB

var MAX_SIZE = 30 * MB //LogN = 15, setup included

var Local = latency.Network{ //simulates LAN on localhost
	Kbps:    1024 * 1024, //1 Gbps
	Latency: 200 * time.Millisecond,
	MTU:     1500, // Ethernet
}

var Lan = latency.Local //no overhead, used in real distributed env.

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
