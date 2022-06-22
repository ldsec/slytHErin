package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
	"sync"
)

//Deals with multipication between encrypted or plaintext encoded block matrices
type Multiplier struct {
	poolSize int
	box      CkksBox
}

//feeded to the workers to tell them what to do
type MulTask struct {
	i, j, k         int
	s               int //k goes from 0 to s
	accumulatorChan chan *ckks.Ciphertext
	//done chan struct{} //flag when accumulator is done
}

func NewMultiplier(Box CkksBox, poolSize int) *Multiplier {
	Mul := new(Multiplier)
	Mul.poolSize = poolSize
	Mul.box = Box
	return Mul
}

func (Mul *Multiplier) spawnEvaluators(X *EncInput, dimIn, dimMid, dimOut int, W BlocksOperand, ch chan MulTask, Out *EncInput) {
	box := BoxShallowCopy(Mul.box)
	for {
		task, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j, k := task.i, task.j, task.k
		ct := DiagMul(X.Blocks[i][k].CopyNew(), dimIn, dimMid, dimOut, W.GetBlock(j, k), true, true, box)
		if k == 0 {
			//I am the accumulator
			defer close(task.accumulatorChan)
			accumulator := 1
			for accumulator < task.s {
				op := <-task.accumulatorChan
				box.Evaluator.Add(ct, op, ct)
				accumulator++
			}
			Out.Blocks[i][j] = ct
		} else {
			//I have to pass
			task.accumulatorChan <- ct
		}
	}
}

//Multiplication between encrypted input and plaintext weight
func (Mul *Multiplier) Multiply(X *EncInput, W BlocksOperand, Box CkksBox) *EncInput {
	//multiplies 2 block matrices, one is encrypted(input) and one not (weight)
	wRowP, wColP := W.GetPartitions()
	dimIn := X.InnerRows
	dimMid, dimOut := W.GetInnerDims()
	//W is block-transposed
	if X.ColP != wColP {
		panic(errors.New("Block partitions not compatible for multiplication"))
	}
	if X.InnerCols != dimMid {
		panic(errors.New("Inner dimentions not compatible for multiplication"))
	}
	q := X.RowP
	//r and s are swapped cause W is block-transposed
	r := wRowP
	s := wColP

	Out := new(EncInput)
	Out.RowP = X.RowP
	Out.ColP = wRowP
	Out.InnerRows = dimIn
	Out.InnerCols = dimOut
	Out.Blocks = make([][]*ckks.Ciphertext, q)
	for i := range Out.Blocks {
		Out.Blocks[i] = make([]*ckks.Ciphertext, r)
	}

	if Mul.poolSize == 1 {
		//single thread
		for i := 0; i < q; i++ {
			for j := 0; j < r; j++ {
				res := new(ckks.Ciphertext)
				for k := 0; k < s; k++ {
					x := X.Blocks[i][k].CopyNew()
					w := W.GetBlock(j, k)
					ct := DiagMul(x, dimIn, dimMid, dimOut, w, true, true, Box)
					if k == 0 {
						res = ct
					} else {
						Box.Evaluator.Add(res, ct, res)
					}
				}
				Out.Blocks[i][j] = res
			}
		}
	} else if Mul.poolSize > 1 {
		//bounded threading
		ch := make(chan MulTask)
		var wg sync.WaitGroup
		//spawn consumers
		for i := 0; i < Mul.poolSize; i++ {
			wg.Add(1)
			go func() {
				Mul.spawnEvaluators(X, dimIn, dimMid, dimOut, W, ch, Out)
				defer wg.Done()
			}()
		}
		//feed consumers
		for i := 0; i < q; i++ {
			for j := 0; j < r; j++ {
				accumulatorChan := make(chan *ckks.Ciphertext, s)
				for k := 0; k < s; k++ {
					task := MulTask{
						i:               i,
						j:               j,
						k:               k,
						s:               s,
						accumulatorChan: accumulatorChan,
					}
					ch <- task
				}
			}
		}
		close(ch)
		wg.Wait()
	}
	return Out
}

//  ---------------------------------------------
//	Operations between encrypted matrices of data
//  |
//  | version with optimized dimentions
//  v

func DiagMul(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []ckks.Operand, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {

	params := Box.Params
	eval := Box.Evaluator

	// Pack value for complex dot-product
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		img := eval.MultByiNew(input)
		eval.Rotate(img, dimIn, img)
		eval.Add(input, img, input)
		replicaFactor := GetReplicaFactor(dimMid, dimOut)
		eval.ReplicateLog(input, dimIn*dimMid, replicaFactor, input)
	}

	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(input, weights[0])

	inputRot := ckks.NewCiphertext(params, 1, input.Level(), input.Scale)

	eval.GetKeySwitcher().DecomposeNTT(input.Level(), params.PCount()-1, params.PCount(), input.Value[1], eval.GetKeySwitcher().BuffDecompQP)

	for i := 1; i < len(weights); i++ {

		eval.PermuteNTTHoisted(input.Level(), input.Value[0], input.Value[1], eval.GetKeySwitcher().BuffDecompQP, 2*dimIn*i, inputRot.Value[0], inputRot.Value[1])

		eval.MulAndAdd(inputRot, weights[i], res)

	}

	// Rescale
	if res.Degree() > 1 {
		eval.Relinearize(res, res)
	}
	eval.Rescale(res, params.DefaultScale(), res)

	// Erases imaginary part
	if cleanImag {
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return
}

//HELPERS
func GetReplicaFactor(dimMid, dimOut int) int {
	if dimOut > dimMid {
		return plainUtils.Max(int(math.Ceil(float64(dimOut)/float64(dimMid))), 3)
	} else {
		return 2
	}
}
