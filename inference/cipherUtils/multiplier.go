package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
	"sync"
	"time"
)

//Deals with multipication between encrypted or plaintext encoded block matrices
type Multiplier struct {
	poolSize int
}

//feeded to the workers to tell them what to do
type MulTask struct {
	i, j, k         int
	s               int //k goes from 0 to s
	accumulatorChan chan *ckks.Ciphertext
	//done chan struct{} //flag when accumulator is done
}

func NewMultiplier(poolSize int) *Multiplier {
	Mul := new(Multiplier)
	Mul.poolSize = poolSize
	return Mul
}

func (Mul *Multiplier) spawnEvaluators(X BlocksOperand, dimIn, dimMid, dimOut int, prepack bool, W BlocksOperand, ch chan MulTask, Out *EncInput, Box CkksBox) {
	box := BoxShallowCopy(Box)
	for {
		task, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j, k := task.i, task.j, task.k
		ct := new(ckks.Ciphertext)
		x := X.GetBlock(i, k)
		w := W.GetBlock(j, k).(DiagMat)
		switch x.(type) {
		case *ckks.Ciphertext:
			ct = DiagMulCt(x.(*ckks.Ciphertext).CopyNew(), dimIn, dimMid, dimOut, w, prepack, box)
		case *ckks.Plaintext:
			ct = DiagMulPt(x.(*ckks.Plaintext), dimIn, w, box)
		}
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
func (Mul *Multiplier) Multiply(X BlocksOperand, W BlocksOperand, prepack bool, Box CkksBox) *EncInput {
	xRowP, xColP := X.GetPartitions()
	xRealRows, _ := X.GetRealDims()
	_, wRealCols := W.GetRealDims()
	wRowP, wColP := W.GetPartitions()
	dimMid, dimOut := W.GetInnerDims()

	//W is block-transposed
	if xColP != wColP {
		switch X.(type) {
		case *EncInput:
			start := time.Now()
			RepackCols(X.(*EncInput), wColP, Box)
			fmt.Println("Done repack: ", time.Since(start))
		default:
			panic(errors.New("Block matrices not compatible for multiplication"))
		}
	}
	dimIn, xInnerCols := X.GetInnerDims()
	if xInnerCols != dimMid {
		panic(errors.New("Inner dimentions not compatible for multiplication"))
	}
	q := xRowP
	//r and s are swapped cause W is block-transposed
	r := wRowP
	s := wColP

	Out := new(EncInput)
	Out.RowP = xRowP
	Out.ColP = wRowP
	Out.InnerRows = dimIn
	Out.InnerCols = dimOut
	Out.RealRows = xRealRows
	Out.RealCols = wRealCols
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
					ct := new(ckks.Ciphertext)
					x := X.GetBlock(i, k)
					w := W.GetBlock(j, k).(DiagMat)
					switch x.(type) {
					case *ckks.Ciphertext:
						ct = DiagMulCt(x.(*ckks.Ciphertext).CopyNew(), dimIn, dimMid, dimOut, w, prepack, Box)
					case *ckks.Plaintext:
						ct = DiagMulPt(x.(*ckks.Plaintext), dimIn, w, Box)
					}
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
				Mul.spawnEvaluators(X, dimIn, dimMid, dimOut, prepack, W, ch, Out, Box)
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
	Mul.RemoveImagFromBlocks(Out, Box)
	return Out
}

//To be called after multiply. Applies the rescaling and removes garbage from imaginary part of slots (from multiplication algo with complex packing)
func (Mul *Multiplier) RemoveImagFromBlocks(X *EncInput, Box CkksBox) {

	poolCh := make(chan struct{}, Mul.poolSize)

	//init channel
	for i := 0; i < Mul.poolSize; i++ {
		poolCh <- struct{}{}
	}
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			<-poolCh //if not routines are available this is blocking
			go func(i, j int, eval ckks.Evaluator) {
				if X.Blocks[i][j].Degree() > 1 {
					eval.Relinearize(X.Blocks[i][j], X.Blocks[i][j])
				}
				eval.Rescale(X.Blocks[i][j], Box.Params.DefaultScale(), X.Blocks[i][j])
				eval.Add(X.Blocks[i][j], eval.ConjugateNew(X.Blocks[i][j]), X.Blocks[i][j])
				poolCh <- struct{}{} //restore 1 go routine in channel

			}(i, j, Box.Evaluator.ShallowCopy())
		}
	}
	for i := 0; i < Mul.poolSize; i++ {
		<-poolCh //empty channel to ensure all gorutines are done
	}
}

//  ---------------------------------------------
//	Operations between encrypted matrices of data
//  |
//  | version with optimized dimentions
//  v

//Multiplies a ciphertext with a weight matrix in diagonal form: W x A.T
func DiagMulCt(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights DiagMat, prepack bool, Box CkksBox) (res *ckks.Ciphertext) {

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
	diags := weights.GetDiags()

	// Lazy inner-product with hoisted rotations
	res = ckks.NewCiphertext(params, 1, input.Level()-1, input.Scale)

	inputRot := ckks.NewCiphertext(params, 1, input.Level(), input.ScalingFactor())

	eval.GetKeySwitcher().DecomposeNTT(input.Level(), params.PCount()-1, params.PCount(), input.Value[1], eval.GetKeySwitcher().BuffDecompQP)

	for i, d := range diags {
		eval.PermuteNTTHoisted(input.Level(), input.Value[0], input.Value[1], eval.GetKeySwitcher().BuffDecompQP, i, inputRot.Value[0], inputRot.Value[1])
		eval.MulAndAdd(inputRot, d, res)
	}

	// Rescale
	if res.Degree() > 1 {
		eval.Relinearize(res, res)
	}
	//
	//// rescales + erases imaginary part
	//
	//eval.Rescale(res, params.DefaultScale(), res)
	//eval.Add(res, eval.ConjugateNew(res), res)
	//
	return
}

//Multiplies a plaintext with a weight matrix in diagonal form: W x A.T
func DiagMulPt(input *ckks.Plaintext, dimIn int, weights DiagMat, Box CkksBox) (res *ckks.Ciphertext) {

	//params := Box.Params
	eval := Box.Evaluator

	diags := weights.GetDiags()
	// Lazy inner-product with hoisted rotations
	res = ckks.NewCiphertext(Box.Params, 1, input.Level()-1, input.Scale)

	i := 0
	rotations := make([]int, len(diags))
	for k := range diags {
		rotations[i] = k
		i++
	}
	inputRot := RotatePlaintext(input, rotations, Box)

	for i := range rotations {
		eval.MulAndAdd(inputRot[i], diags[i], res)
	}

	// Rescale
	//
	//// rescales + erases imaginary part
	//
	//eval.Rescale(res, params.DefaultScale(), res)
	//eval.Add(res, eval.ConjugateNew(res), res)
	//
	return
}

//Applies complex packing to Blocks. dimOut should be the innerCols of the first weight in the layers
func PrepackBlocks(X BlocksOperand, dimOut int, Box CkksBox) {
	eval := Box.Evaluator
	switch X.(type) {
	case *EncInput:
		Xenc := X.(*EncInput)
		for i := 0; i < Xenc.RowP; i++ {
			for j := 0; j < Xenc.ColP; j++ {
				Prepack(Xenc.Blocks[i][j], Xenc.InnerRows, Xenc.InnerCols, dimOut, eval)
			}
		}
	case *PlainInput:
		//plaintext
		Xp := X.(*PlainInput)
		for i := range Xp.Blocks {
			for j := range Xp.Blocks[i] {
				X.(*PlainInput).Blocks[i][j] = PrepackClearText(Xp.Blocks[i][j], Xp.InnerRows, Xp.InnerCols, dimOut, Box)
			}
		}
	}
}

//Prepacking cipher
func Prepack(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, eval ckks.Evaluator) {
	img := eval.MultByiNew(input)
	eval.Rotate(img, dimIn, img)
	eval.Add(input, img, input)
	replicaFactor := GetReplicaFactor(dimMid, dimOut)
	eval.ReplicateLog(input, dimIn*dimMid, replicaFactor, input)
}

//Prepacking plain
func PrepackClearText(input *ckks.Plaintext, dimIn, dimMid, dimOut int, Box CkksBox) *ckks.Plaintext {
	tmp := Box.Encoder.Decode(input, Box.Params.LogSlots())
	img := plainUtils.MulByi(plainUtils.ComplexToReal(tmp))
	img = plainUtils.RotateComplexArray(img, dimIn)
	for k := range tmp {
		tmp[k] += img[k] //complex packing of cols
	}
	for k, kk := dimIn*(dimMid-1), 0; k < dimIn*dimMid; k, kk = k+1, kk+1 {
		tmp[k] += img[len(img)-dimIn+kk] //add first col into last col
	}
	tmp = plainUtils.ReplicateComplexArray(tmp[:dimIn*dimMid], GetReplicaFactor(dimMid, dimOut))
	return Box.Encoder.EncodeNew(tmp, Box.Params.MaxLevel(), Box.Params.DefaultScale(), Box.Params.LogSlots())
}

//Repacks block matrix column partitions to have newColP = colP. Does not involve multiplication or rescaling
func RepackCols(X *EncInput, colP int, Box CkksBox) {
	cols := X.ColP * X.InnerCols
	if cols%colP != 0 {
		panic(errors.New("Target Partition not compatible with given Block Matrix"))
	}
	if X.InnerRows*(cols/colP)*2 > Box.Params.Slots() {
		panic(errors.New("New inner dimention is too big. Must be <= Slots / 2"))
	}
	if X.ColP == 1 || X.ColP == colP {
		fmt.Println("Repacking: Nothing to do")
		return
	}
	fmt.Println("Repacking...")

	eval := Box.Evaluator

	if X.ColP%colP == 0 {
		// new partition is a divisor of current

		buffer := make([][]*ckks.Ciphertext, X.RowP)
		innerBlocks := X.ColP / colP

		var wg sync.WaitGroup

		for i := 0; i < X.RowP; i++ {
			//for each row, unite blocks
			buffer[i] = make([]*ckks.Ciphertext, colP)
			//fmt.Println("Row ", i)
			for part := 0; part < colP; part++ {
				wg.Add(1)
				go func(i, part int, eval ckks.Evaluator) {
					defer wg.Done()
					accumulator := X.Blocks[i][part*innerBlocks].CopyNew()
					//fmt.Println("Accumulator is at ", i, " - ", part*innerBlocks, "up to ", part*innerBlocks+innerBlocks)
					for j := part*innerBlocks + 1; j < part*innerBlocks+innerBlocks; j++ {
						eval.Add(accumulator, eval.RotateNew(X.Blocks[i][j], -X.InnerRows*X.InnerCols*(j%innerBlocks)), accumulator)
					}
					buffer[i][part] = accumulator
				}(i, part, eval.ShallowCopy())
			}
		}
		wg.Wait()

		X.Blocks = buffer
		X.ColP = colP
		X.InnerCols = cols / colP
		return

	} else {
		//new partition is not a divisor of current partition
		var wg sync.WaitGroup

		blocks := make([][]*ckks.Ciphertext, X.RowP)
		newInnerCols := (X.InnerCols * X.ColP) / colP

		mask := make([]float64, X.InnerRows*newInnerCols)
		for i := range mask {
			mask[i] = 1.0
		}
		maskEcd := Box.Encoder.EncodeNew(mask, X.Blocks[0][0].Level(), Box.Params.QiFloat64(X.Blocks[0][0].Level()), Box.Params.LogSlots())

		for i := 0; i < X.RowP; i++ {
			blocks[i] = make([]*ckks.Ciphertext, colP)
			buffer := X.Blocks[i][0]
			buffered := X.InnerCols
			filled := buffered
			completedBlocks := 0
			j := 1

			for completedBlocks < colP {
				if filled < newInnerCols {
					if j < X.ColP {
						wg.Add(1)
						go func(i, j, completedBlocks int, eval ckks.Evaluator, buffer *ckks.Ciphertext, filled int) {

							defer wg.Done()
							blocks[i][completedBlocks] = eval.AddNew(buffer, eval.RotateNew(X.Blocks[i][j], -filled*X.InnerRows))
							//cleanup
							eval.Mul(blocks[i][completedBlocks], maskEcd, blocks[i][completedBlocks])
							eval.Rescale(blocks[i][completedBlocks], X.Blocks[0][0].Scale, blocks[i][completedBlocks])
						}(i, j, completedBlocks, eval.ShallowCopy(), buffer.CopyNew(), filled)

						completedBlocks++
						if filled != 0 {
							eval.Rotate(X.Blocks[i][j], (newInnerCols-filled)*X.InnerRows, buffer)
							buffered = X.InnerCols - (newInnerCols - filled)
							filled = buffered
							j++
						}
						if buffered == 0 && j+1 < X.ColP {
							//buffer next block if available
							buffer = X.Blocks[i][j]
							buffered = X.InnerCols
							filled = buffered
							j++
						}
					}
				}
			}
		}
		wg.Wait()
		X.Blocks = blocks
		X.ColP = colP
		X.InnerCols = cols / colP
		return
	}
}

//HELPERS

//Gets the replication factor used for the multipication algorithm given the inner rows and cols of the weight block-matrix
func GetReplicaFactor(dimMid, dimOut int) int {
	if dimOut > dimMid {
		return plainUtils.Max(int(math.Ceil(float64(dimOut)/float64(dimMid)))+1, 3)
	} else {
		return 2
	}
}

//returns array of Plaintexts, where ith plaintext is rotated by rot[i] to the left
func RotatePlaintext(pt *ckks.Plaintext, rotations []int, box CkksBox) []*ckks.Plaintext {
	ptRot := make([]*ckks.Plaintext, len(rotations))
	for i, rot := range rotations {
		tmp := box.Encoder.Decode(pt, box.Params.LogSlots())
		tmp = plainUtils.RotateComplexArray(tmp, rot)
		ptRot[i] = box.Encoder.EncodeNew(tmp, pt.Level(), pt.Scale, box.Params.LogSlots())
	}
	return ptRot
}

func GenRotationsForRepackCols(innerR, currCols, innerC, newColP int) []int {
	var rotations []int
	for i := 1; i < newColP; i++ {
		rotations = append(rotations, -innerR*innerC*i)
	}
	newInnerC := currCols / newColP
	for i := 1; i < newInnerC; i++ {
		rotations = append(rotations, -i*innerR)
		rotations = append(rotations, (newInnerC-i)*innerR)
	}
	return rotations
}
