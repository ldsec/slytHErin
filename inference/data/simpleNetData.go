package data

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

type DataSimpleNet struct {
	X             [][]float64 `json:"X"`
	Y             []int       `json:"Y"`
	InputLayerDim int         //dimention of first layer of SimpleNet
	BatchSize     int
	NumBatches    int
	CurrentBatch  int
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
func (data *DataSimpleNet) Init(batchSize int, inputLayerDim int) error {
	if 2*batchSize*inputLayerDim > 1<<14 { //fixed number of slots per formatted input
		return errors.New("Batch too big for encryption")
	}
	data.BatchSize = batchSize
	data.InputLayerDim = inputLayerDim
	totData := len(data.Y)
	data.NumBatches = int(math.Floor(float64(totData) / float64(batchSize)))
	data.CurrentBatch = 0
	return nil
}

func (data *DataSimpleNet) Batch() ([][]float64, []int, error) {
	if data.CurrentBatch < data.NumBatches {
		i := data.CurrentBatch * data.BatchSize
		j := (data.CurrentBatch + 1) * data.BatchSize
		Xbatch := data.X[i:j]
		//add padding
		Xpad := make([][]float64, data.BatchSize)
		for k := 0; k < data.BatchSize; k++ {
			Xpad[k] = make([]float64, data.InputLayerDim)
			for z := 0; z < len(Xbatch[k]); z++ {
				Xpad[k][z] = Xbatch[k][z]
			}
		}
		Y := data.Y[i:j]
		data.CurrentBatch += 1
		return Xpad, Y, nil
	}
	//last batch is incomplete
	return nil, nil, errors.New("No more complete batches")
}
