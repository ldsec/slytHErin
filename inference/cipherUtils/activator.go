//Package contains the logic for all operations between ciphertext and plaintexts
package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

//Wrapper for polynomial activation function
type activationPoly struct {
	polynomial  *ckks.Polynomial
	term0VecEcd *ckks.Plaintext
}

//Handles an activation layer with a polynomial function on an EncInput
type Activator struct {
	poly             []activationPoly
	box              CkksBox //pool of evaluators to be used
	isCheby          bool
	poolSize         int
	NumOfActivations int
}

type EvalFunc func(X *EncInput, i, j int, poly activationPoly, Box CkksBox)

//Creates a new Activator. Takes lavel, scale, as well as the blocks and sub-matrices dimentions, of the
//output of the previous linear layer
func NewActivator(numOfActivations int, Box CkksBox, poolSize int) (*Activator, error) {
	Act := new(Activator)
	Act.poolSize = poolSize
	Act.NumOfActivations = numOfActivations
	Act.poly = make([]activationPoly, Act.NumOfActivations)
	Act.box = Box
	return Act, nil
}

//Add activation functions at layer. Takes level and scale of ct to activate at layer, as well its inner dimentions
func (Act *Activator) AddActivation(activation utils.ChebyPolyApprox, layer, level int, scale float64, innerRows, innerCols int) {
	poly := new(ckks.Polynomial)
	i := layer
	Box := Act.box
	switch activation.ChebyBase {
	case false:
		//if not cheby base we extract the const term and add it later so to use a more efficient poly eval without encoder
		term0 := activation.Poly.Coeffs[0]
		poly = activation.Poly
		Act.isCheby = false
		term0vec := make([]complex128, innerRows*innerCols)
		for i := range term0vec {
			term0vec[i] = term0
		}
		Act.poly[i] = activationPoly{}
		if poly.Depth() > level {
			panic(errors.New("Not enough levels for poly depth"))
		}
		Act.poly[i].term0VecEcd = ckks.NewPlaintext(Box.Params, level-poly.Depth(), scale)
		Box.Encoder.EncodeSlots(term0vec, Act.poly[i].term0VecEcd, Box.Params.LogSlots())

		//copy the polynomial with the 0 const term without affecting the original poly
		polyCp := new(ckks.Polynomial)
		*polyCp = *poly
		polyCp.Coeffs = make([]complex128, len(poly.Coeffs))
		copy(polyCp.Coeffs, poly.Coeffs)
		polyCp.Coeffs[0] = complex(0, 0)
		Act.poly[i].polynomial = polyCp

	case true:
		Act.isCheby = true
		Act.poly[i] = activationPoly{}
		Act.poly[i].polynomial = activation.Poly
	default:
		panic(errors.New("Activation of Activator is not a known type"))
	}
}

// returns levels needed for activation at layer
func (Act *Activator) LevelsOfAct(layer int) int {
	return Act.poly[layer].polynomial.Depth()
}

func (Act *Activator) spawnEvaluators(f EvalFunc, poly activationPoly, X *EncInput, ch chan []int) {
	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		f(X, i, j, poly, Act.box)
	}
}

//Evaluates a polynomial on the ciphertext. If no activation is at layer, applies identity
func (Act *Activator) ActivateBlocks(X *EncInput, layer int) {
	if layer >= Act.NumOfActivations {
		//no more act -> identity
		return
	}
	var f EvalFunc
	if !Act.isCheby {
		f = evalPolyBlocks
	} else {
		f = evalPolyBlocksVector
	}
	if Act.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				f(X, i, j, Act.poly[layer], Act.box)
			}
		}
	} else if Act.poolSize > 1 {
		//bounded threading
		ch := make(chan []int)
		var wg sync.WaitGroup
		for i := 0; i < Act.poolSize; i++ {
			wg.Add(1)
			go func() {
				Act.spawnEvaluators(f, Act.poly[layer], X, ch)
				defer wg.Done()
			}()
		}
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				//feed consumers
				ch <- []int{i, j}
			}
		}
		close(ch)
		wg.Wait()
	}
}

//Evaluates a polynomial on the ciphertext
func evalPolyBlocks(X *EncInput, i, j int, poly activationPoly, Box CkksBox) {
	eval := Box.Evaluator.ShallowCopy()
	ct, _ := eval.EvaluatePoly(X.Blocks[i][j], poly.polynomial, X.Blocks[i][j].Scale)
	eval.Add(ct, poly.term0VecEcd, ct)
	X.Blocks[i][j] = ct
}

//Evaluates a polynomial on the ciphertext using ckks.Evaluator.EvaluatePolyVector. This should be called via Activator.ActivateBlocks
func evalPolyBlocksVector(X *EncInput, i, j int, poly activationPoly, Box CkksBox) {
	//build map of all slots with legit values
	eval := Box.Evaluator.ShallowCopy()
	ecd := Box.Encoder.ShallowCopy()
	slotsIndex := make(map[int][]int)
	idx := make([]int, X.InnerRows*X.InnerCols)
	for i := 0; i < X.InnerRows*X.InnerCols; i++ {
		idx[i] = i
	}
	slotsIndex[0] = idx
	ct, _ := eval.EvaluatePolyVector(X.Blocks[i][j], []*ckks.Polynomial{poly.polynomial}, ecd, slotsIndex, X.Blocks[i][j].Scale)
	X.Blocks[i][j] = ct
}
