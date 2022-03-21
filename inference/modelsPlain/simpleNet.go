package modelsPlain

import (
	//"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type Bias struct {
	B   []float64 `json:"b"`
	Len int       `json:"len"`
}
type Channel struct {
	W    []float64 `json:"w"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}
type Kernel struct {
	Channels []Channel `json:"channels"`
}

type ConvLayer struct {
	Weight []Kernel `json:"weight"`
	Bias   Bias     `json:"bias"`
	//kernelSize, inChans, outChans, stride, inDim, outDim int
}
type SimpleNet struct {
	Conv1 ConvLayer `json:"conv1"`
	Pool1 ConvLayer `json:"pool1"`
	Pool2 ConvLayer `json:"pool2"`

	//reluApprox ckks.Polynomial //this will store the coefficients of the poly approximating ReLU
}

func LoadSimpleNet(path string) SimpleNet {
	//jsonFile, err := os.Open("../../training/models/simpleNet.json")
	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res SimpleNet
	json.Unmarshal([]byte(byteValue), &res)
	fmt.Println(res.Conv1)
	return res
}
