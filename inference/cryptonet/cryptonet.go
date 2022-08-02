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

//json wrapper

type CryptoNet struct {
	network.Network
}

type CryptoNetHE struct {
	*network.HENetwork
}

func (cn *CryptoNet) InitActivations() []utils.ChebyPolyApprox {
	approx := utils.InitReLU(3)
	return []utils.ChebyPolyApprox{*approx, *approx}
}

func (l *CNLoader) Load(path string) network.NetworkI {

	jsonFile, err := os.Open(path)
	utils.ThrowErr(err)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	nj := new(network.NetworkJ)
	err = json.Unmarshal([]byte(byteValue), nj)
	utils.ThrowErr(err)
	cn := new(CryptoNet)
	cn.SetLayers(nj.Layers)
	activations := cn.InitActivations()
	cn.SetActivations(activations)

	//return cn
	return nil
}

func (cn *CryptoNet) NewCryptoNet(splits []cipherUtils.BlockSplits, encrypted, bootstrappable bool, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) *CryptoNetHE {
	return &CryptoNetHE{cn.NewHE(splits, encrypted, bootstrappable, minLevel, btpCapacity, Bootstrapper, poolsize, Box).(*network.HENetwork)}
}
