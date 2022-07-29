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

//json wrapper
type nn struct {
	Conv   utils.Layer   `json:"conv"`
	Dense  []utils.Layer `json:"dense"`
	Layers int           `json:"layers"`
}

type NN struct {
	network.Network
}

type NNHE struct {
	*network.HENetwork
}

//Initialize activation function. Needs path of json file of intervals
func (nn *NN) InitActivations(layers int, HEtrain bool) []utils.ChebyPolyApprox {
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

func (l *NNLoader) Load(path string) network.NetworkI {
	jsonFile, err := os.Open(path)
	if err != nil {
		utils.ThrowErr(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res nn
	json.Unmarshal([]byte(byteValue), &res)

	layers := []utils.Layer{res.Conv}
	layers = append(layers, res.Dense...)

	nn := new(NN)
	HEtrain := strings.Contains(path, "poly")

	activations := nn.InitActivations(res.Layers, HEtrain)
	nn.SetLayers(layers)
	nn.SetActivations(activations)
	return nn
}

func (nn *NN) NewNN(splits []cipherUtils.BlockSplits, encrypted, bootstrappable bool, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) *NNHE {
	return &NNHE{nn.NewHE(splits, encrypted, bootstrappable, minLevel, btpCapacity, Bootstrapper, poolsize, Box).(*network.HENetwork)}
}
