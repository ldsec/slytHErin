package data

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

type Data struct {
	X            [][]float64 `json:"X"`
	Y            []int       `json:"Y"`
	BatchSize    int
	NumBatches   int
	CurrentBatch int
}

func LoadData(path string) *Data {
	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res Data
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}
func (data *Data) Init(batchSize int) error {
	data.BatchSize = batchSize
	totData := len(data.Y)
	data.NumBatches = int(math.Floor(float64(totData) / float64(batchSize)))
	data.CurrentBatch = 0
	return nil
}

func (data *Data) Batch() ([][]float64, []int, error) {
	if data.CurrentBatch < data.NumBatches {
		i := data.CurrentBatch * data.BatchSize
		j := (data.CurrentBatch + 1) * data.BatchSize
		Xbatch := data.X[i:j]
		Y := data.Y[i:j]
		data.CurrentBatch += 1
		return Xbatch, Y, nil
	}
	//last batch is incomplete
	return nil, nil, errors.New("No more complete batches")
}
