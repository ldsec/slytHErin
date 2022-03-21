package main

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/modelsPlain"
)

func main() {
	sn := modelsPlain.LoadSimpleNet("../training/models/simpleNet2.json")
	fmt.Println(sn)
}
