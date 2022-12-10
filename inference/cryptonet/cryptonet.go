//Contains declaration and experiments for CryptoNet model
package cryptonet

import (
	"encoding/json"
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

func InitActivations(args ...interface{}) []utils.ChebyPolyApprox {
	approx := utils.InitReLU(3)
	return []utils.ChebyPolyApprox{*approx, *approx}
}

func (l *CNLoader) Load(path string, initActivations network.Initiator) network.NetworkI {

	jsonFile, err := os.Open(path)
	utils.ThrowErr(err)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	nj := new(network.NetworkJ)
	err = json.Unmarshal([]byte(byteValue), nj)
	utils.ThrowErr(err)
	cn := new(CryptoNet)
	cn.SetLayers(nj.Layers)
	activations := initActivations()
	cn.SetActivations(activations)

	return cn
}
