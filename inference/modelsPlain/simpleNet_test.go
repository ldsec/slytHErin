package modelsPlain

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/data"
	"testing"
)

func evalPlain_test(t *testing.T) {
	sn := modelsPlain.LoadSimpleNet("../../training/models/simpleNet2.json")
	fmt.Println(sn)
	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")

}
