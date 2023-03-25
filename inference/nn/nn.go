//definitions and experiments for zama nn models
package nn

import (
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/network"
	"github.com/ldsec/dnn-inference/inference/utils"
	"io/ioutil"
	"os"
	"strings"
)

type NNLoader struct {
	network.NetworkLoader
}

func (l *NNLoader) IsInit(network network.NetworkI) bool {
	return network.IsInit()
}

type NN struct {
	network.Network
}

type NNHE struct {
	*network.HENetwork
}

var DEG = 63

//Initialize activation function
func InitActivations(args ...interface{}) []utils.ChebyPolyApprox {
	layers := args[0].(int)
	HEtrain := args[1].(bool)
	var suffix string
	var act string
	activations := make([]utils.ChebyPolyApprox, layers)
	if HEtrain {
		suffix = "_poly"
		act = "silu"
	} else {
		suffix = ""
		act = "soft relu"
	}
	jsonFile, err := os.Open(fmt.Sprintf("nn%d%s_intervals.json", layers, suffix))
	utils.ThrowErr(err)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)
	var intervals utils.ApproxParams
	json.Unmarshal([]byte(byteValue), &intervals)
	intervals = utils.SetDegOfParam(intervals)
	for i := range intervals.Params {
		interval := intervals.Params[i]
		activations[i] = *utils.InitActivationCheby(act, interval.A, interval.B, interval.Deg)
	}
	return activations
}

func (l *NNLoader) Load(path string, initActivations network.Initiator) network.NetworkI {
	jsonFile, err := os.Open(path)
	if err != nil {
		utils.ThrowErr(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	nj := new(network.NetworkJ)
	utils.ThrowErr(json.Unmarshal([]byte(byteValue), nj))

	HEtrain := strings.Contains(path, "poly")

	nn := new(NN)
	nn.SetLayers(nj.Layers)
	activations := initActivations(nn.GetNumOfLayers()-1, HEtrain)
	nn.SetActivations(activations)
	return nn
}

func (nn *NN) NewNN(splits *cipherUtils.Split, encrypted, bootstrappable bool, maxLevel, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) *NNHE {
	return &NNHE{nn.NewHE(splits, encrypted, bootstrappable, maxLevel, minLevel, btpCapacity, Bootstrapper, poolsize, Box).(*network.HENetwork)}
}
