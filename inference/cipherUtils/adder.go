package cipherUtils

import (
	"errors"
	"sync"
)

//Deals with addition between encrypted or plaintext encoded block matrices
type Adder struct {
	poolSize int
	B        BlocksOperand //either plaintext or ecnrypted bias
	box      CkksBox
}

func NewAdder(B BlocksOperand, Box CkksBox, poolSize int) *Adder {
	switch B.(type) {
	case *EncInput:
	case *PlainInput:
	default:
		panic(errors.New("Adder supports either *EncInput or *PlainInput"))
	}
	Ad := new(Adder)
	Ad.B = B
	Ad.poolSize = poolSize
	Ad.box = Box
	return Ad
}

func (Ad *Adder) spawnEvaluators(X *EncInput, ch chan []int) {
	eval := Ad.box.Evaluator.ShallowCopy()

	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		X.Blocks[i][j] = eval.AddNew(X.Blocks[i][j], Ad.B.GetBlock(i, j)[0])
	}
}

//Addition between encrypted input and bias. This modifies X
func (Ad *Adder) AddBias(X *EncInput) {
	rowP, colP := Ad.B.GetPartitions()
	if X.RowP != rowP || X.ColP != colP {
		panic(errors.New("Block partitions not compatible for addition"))
	}
	innerR, innerC := Ad.B.GetInnerDims()
	if X.InnerRows != innerR || X.InnerCols != innerC {
		panic(errors.New("Inner dimentions not compatible for addition"))
	}
	if Ad.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				X.Blocks[i][j] = Ad.box.Evaluator.AddNew(X.Blocks[i][j], Ad.B.GetBlock(i, j)[0])
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
				Ad.spawnEvaluators(X, ch)
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
