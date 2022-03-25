package modelsPlain

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/data"
	"testing"
)

func TestEvalPlain(t *testing.T) {
	sn := LoadSimpleNet("../../training/models/simpleNet.json")
	sn.InitDim()
	sn.InitActivation()
	batchSize := 64
	dataSn := data.LoadSimpleNetData("../../training/data/simpleNet_data.json")
	dataSn.Init(batchSize)
	corrects := 0
	tot := 0
	for true {
		Xbatch, Y, err := dataSn.Batch()
		if err != nil {
			break
		}
		corrects += sn.EvalBatchPlain(Xbatch, Y)
		tot += batchSize
	}
	fmt.Println("Accuracy:", float64(corrects)/float64(tot))
}
