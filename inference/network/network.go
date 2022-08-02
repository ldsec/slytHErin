package network

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
)

//Custom Network Loader.
//Exposes the method Load to load model from file
//User should take care of defining a correct struct to parse the json file containing the network
//The only constraint is that
type NetworkLoader interface {
	Load(path string) NetworkI
}

//Network loaded from json
type NetworkI interface {
	SetBatch(batchSize int)
	SetLayers(layers []utils.Layer)
	SetActivations(activations []utils.ChebyPolyApprox)
	//returns weights and biases
	GetParams() ([]*mat.Dense, []*mat.Dense)
	//returns weight and biases rescaled for approximated activation
	GetParamsRescaled() ([]*mat.Dense, []*mat.Dense)
	GetActivations() []utils.ChebyPolyApprox
	GetNumOfLayers() int
	GetNumOfActivations() int
	GetDimentions() ([]int, []int)
	//true if network is initialized, i.e if batch is defined as well as layers and activations
	IsInit() bool
	//true if he version needs bootstrapping at this layer with level = level
	CheckLvlAtLayer(level, minLevel, layer int, forAct, afterMul bool) bool
	//computes how many levels are needed to complete the pipeline in he version
	LevelsToComplete(currLayer int, afterMul bool) int
	//Creates the HE version of this network
	NewHE(splits []cipherUtils.BlockSplits, encrypted, bootstrappable bool, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) HENetworkI
}

//Network loaded from json. Implements INetwork. Abstract type
type Network struct {
	layers           []utils.Layer
	activations      []utils.ChebyPolyApprox
	numOfLayers      int
	numOfActivations int
	batchSize        int
}

func (n *Network) SetLayers(layers []utils.Layer) {
	n.layers = layers
	n.numOfLayers = len(layers)
}

func (n *Network) SetActivations(activations []utils.ChebyPolyApprox) {
	n.activations = activations
	n.numOfActivations = len(activations)
}

func (n *Network) GetActivations() []utils.ChebyPolyApprox {
	return n.activations
}

func (n *Network) GetNumOfLayers() int {
	return n.numOfLayers
}

func (n *Network) GetNumOfActivations() int {
	return n.numOfActivations
}

func (n *Network) GetParams() ([]*mat.Dense, []*mat.Dense) {
	if !n.IsInit() {
		panic(errors.New("Not init"))
	}
	w := make([]*mat.Dense, n.numOfLayers)
	b := make([]*mat.Dense, n.numOfLayers)
	for i := range w {
		w[i], b[i] = n.layers[i].Build(n.batchSize)
	}
	return w, b
}

func (n *Network) getWeights() []*mat.Dense {
	if n.layers == nil {
		panic(errors.New("Layers not set"))
	}
	w := make([]*mat.Dense, n.numOfLayers)
	for i := range w {
		w[i] = n.layers[i].BuildWeight()
	}
	return w
}

func (n *Network) SetBatch(batchSize int) {
	n.batchSize = batchSize
}

func (n *Network) GetParamsRescaled() ([]*mat.Dense, []*mat.Dense) {
	if !n.IsInit() {
		panic("Not init")
	}
	scaledW := make([]*mat.Dense, len(n.layers))
	scaledB := make([]*mat.Dense, len(n.layers))
	for i, l := range n.layers {
		//skip last as no activation
		w, b := l.Build(n.batchSize)
		if i < n.numOfActivations {
			scaledW[i], scaledB[i] = n.activations[i].Rescale(w, b)
		} else {
			scaledW[i], scaledB[i] = w, b
		}
	}
	return scaledW, scaledB
}

func (n *Network) LevelsToComplete(currLayer int, afterMul bool) int {
	if !n.IsInit() {
		panic(errors.New("Not Inited!"))
	}
	levelsNeeded := 0
	for i := currLayer; i < n.numOfLayers; i++ {
		levelsNeeded += 1 //mul
		if i < n.numOfActivations {
			//last layer with no act
			levelsNeeded += n.activations[i].LevelsOfAct()
		}
	}
	if afterMul {
		levelsNeeded--
	}
	//fmt.Printf("Levels needed from layer %d to complete: %d\n\n", currLayer+1, levelsNeeded)
	return levelsNeeded
}

func (n *Network) CheckLvlAtLayer(level, minLevel, layer int, forAct, afterMul bool) bool {
	if !n.IsInit() {
		panic(errors.New("Not Inited!"))
	}
	levelsOfAct := 0
	if layer < n.GetNumOfActivations() && forAct {
		levelsOfAct = n.GetActivations()[layer].LevelsOfAct()
	}
	return (level < levelsOfAct || level <= minLevel || level-levelsOfAct < minLevel) && level < n.LevelsToComplete(layer, afterMul)
}

//Returns array of rows and cols of weights
func (n *Network) GetDimentions() ([]int, []int) {
	w := n.getWeights()
	rows, cols := make([]int, len(w)), make([]int, len(w))
	for i := range rows {
		rows[i], cols[i] = w[i].Dims()
	}
	return rows, cols
}

func (n *Network) IsInit() bool {
	return n.batchSize != 0 && n.layers != nil && n.activations != nil
}

func (n *Network) NewHE(splits []cipherUtils.BlockSplits, encrypted, bootstrappable bool, minLevel, btpCapacity int, Bootstrapper cipherUtils.IBootstrapper, poolsize int, Box cipherUtils.CkksBox) HENetworkI {
	nnhe := NewHENetwork(n, splits, encrypted, bootstrappable, minLevel, btpCapacity, Bootstrapper, poolsize, Box)
	return nnhe
}