package cipherUtils

import (
	"errors"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

//Deals with addition between encrypted or plaintext encoded block matrices
type Adder struct {
	poolSize int
}

func NewAdder(poolSize int) *Adder {
	Ad := new(Adder)
	Ad.poolSize = poolSize
	return Ad
}

func (Ad *Adder) spawnEvaluators(X *EncInput, B BlocksOperand, ch chan []int, Box CkksBox) {
	eval := Box.Evaluator.ShallowCopy()
	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		X.Blocks[i][j] = eval.AddNew(X.Blocks[i][j], B.GetBlock(i, j).(ckks.Operand))
	}
}

//Addition between encrypted input and bias. This modifies X
func (Ad *Adder) AddBias(X *EncInput, B BlocksOperand, Box CkksBox) {
	rowP, colP := B.GetPartitions()
	if X.RowP != rowP || X.ColP != colP {
		panic(errors.New("Block partitions not compatible for addition"))
	}
	innerR, innerC := B.GetInnerDims()
	if X.InnerRows != innerR || X.InnerCols != innerC {
		panic(errors.New("Inner dimentions not compatible for addition"))
	}
	if Ad.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				X.Blocks[i][j] = Box.Evaluator.AddNew(X.Blocks[i][j], B.GetBlock(i, j).(ckks.Operand))
			}
		}
	} else if Ad.poolSize > 1 {
		//bounded threading

		ch := make(chan []int)
		var wg sync.WaitGroup
		//spawn consumers
		for i := 0; i < Ad.poolSize; i++ {
			wg.Add(1)
			go func() {
				Ad.spawnEvaluators(X, B, ch, Box)
				defer wg.Done()
			}()
		}
		//feed consumers
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				ch <- []int{i, j}
			}
		}
		close(ch)
		wg.Wait()
	}
}
