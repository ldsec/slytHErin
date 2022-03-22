package data

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type DataSimpleNet struct {
	X [][]float64 `json:"X"`
	Y [][]float64 `json:"Y"`
}

func LoadSimpleNetData(path string) *DataSimpleNet {
	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res DataSimpleNet
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}
