package modelsPlain

import (
	"github.com/tuneinsight/lattigo/v3/ckks"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)
type weightOfConv1 struct{

}

type conv1 struct{

}
type SimpleNet struct {


	reluApprox ckks.Polynomial //this will store the coefficients of the poly approximating ReLU
}

func (sn *SimpleNet) LoadWeights(path string) {
	//jsonFile, err := os.Open("../../training/models/simpleNet.json")
	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var result map[string]interface{}
	json.Unmarshal([]byte(byteValue), &result)

	fmt.Println(result)
}