package modelsPlain

import (
	"fmt"
	"testing"
)

func testLoad(t *testing.T){
	sm := new(SimpleNet)
	fmt.Println("a")
	sm.LoadWeights("../../training/models/simpleNet.json")
}