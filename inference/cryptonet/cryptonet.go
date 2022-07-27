package cryptonet

import (
	"encoding/json"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/network"
	"github.com/ldsec/dnn-inference/inference/utils"
	"io/ioutil"
	"os"
)

type CNLoader struct {
	network.NetworkLoader
}

func (l *CNLoader) IsInit(network network.NetworkI) bool {
	return network.IsInit()
}

//json wrapper
type cryptonet struct {
	Conv1 utils.Layer `json:"conv1"`
	Pool1 utils.Layer `json:"pool1"`
	Pool2 utils.Layer `json:"pool2"`
}

type CryptoNet struct {
	network.Network
}

type CryptoNetHE struct {
	*network.HENetwork
}

func (cn *CryptoNet) InitActivations() []utils.ChebyPolyApprox {
	approx := utils.InitReLU(3)
	return []utils.ChebyPolyApprox{*approx}
}

func (l *CNLoader) Load(path string) network.NetworkI {

	jsonFile, err := os.Open(path)
	utils.ThrowErr(err)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res cryptonet
	json.Unmarshal([]byte(byteValue), &res)

	layers := []utils.Layer{res.Conv1, res.Pool1, res.Pool2}

	cn := new(CryptoNet)
	activations := cn.InitActivations()
	cn.SetActivations(activations)
	cn.SetLayers(layers)

	return cn
}

func (cn *CryptoNet) NewCryptoNet(splits []cipherUtils.BlockSplits, encrypted, bootstrappable bool, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) *CryptoNetHE {
	return &CryptoNetHE{cn.NewHE(splits, encrypted, bootstrappable, minLevel, btpCapacity, Bootstrapper, poolsize, Box).(*network.HENetwork)}
}
