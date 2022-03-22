package main

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/data"
)

func main() {
	//sn := modelsPlain.LoadSimpleNet("../training/models/simpleNet2.json")
	//fmt.Println(sn)
	dataSn := data.LoadSimpleNetData("../training/data/simpleNet_data.json")
	fmt.Println(dataSn.X[0], dataSn.Y[0])
}
