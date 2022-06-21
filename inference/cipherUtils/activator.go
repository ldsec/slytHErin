package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

//Handles an activation layer with a polynomial function on an EncInput

type ActivationPoly struct {
	polynomial  *ckks.Polynomial
	term0VecEcd *ckks.Plaintext
}

type Activator struct {
	poly     ActivationPoly
	box      CkksBox //pool of evaluators to be used
	isCheby  bool
	poolSize int
}

type EvalFunc func(X *EncInput, i, j int, poly ActivationPoly, Box CkksBox)

//Creates a new Activator. Takes lavel, scale, as well as the blocks and sub-matrices dimentions, of the
//output of the previous linear layer
func NewActivator(activation interface{}, level int, scale float64, innerRows, innerCols int, Box CkksBox, poolSize int) (*Activator, error) {
	Act := new(Activator)
	Act.poolSize = poolSize
	poly := new(ckks.Polynomial)
	switch activation.(type) {
	case *utils.MinMaxPolyApprox:
		//if poly is MinMax Approx we extract the const term and add it later so to use a more efficient poly eval without encoder
		term0 := activation.(*utils.MinMaxPolyApprox).Poly.Coeffs[0]
		poly = activation.(*utils.MinMaxPolyApprox).Poly
		Act.isCheby = false
		term0vec := make([]complex128, innerRows*innerCols)
		for i := range term0vec {
			term0vec[i] = term0
		}
		Act.poly = ActivationPoly{}
		if poly.Depth() > level {
			return nil, errors.New("Not enough levels for poly depth")
		}
		Act.poly.term0VecEcd = ckks.NewPlaintext(Box.Params, level-poly.Depth(), scale)
		Box.Encoder.EncodeSlots(term0vec, Act.poly.term0VecEcd, Box.Params.LogSlots())

		//copy the polynomial with the 0 const term without affecting the original poly
		polyCp := new(ckks.Polynomial)
		*polyCp = *poly
		polyCp.Coeffs = make([]complex128, len(poly.Coeffs))
		copy(polyCp.Coeffs, poly.Coeffs)
		polyCp.Coeffs[0] = complex(0, 0)
		Act.poly.polynomial = polyCp

		Act.box = Box
		return Act, nil

	case *utils.ChebyPolyApprox:
		Act.isCheby = true
		Act.poly = ActivationPoly{}
		Act.poly.polynomial = activation.(*utils.ChebyPolyApprox).Poly
		Act.box = Box
		return Act, nil
	default:
		return nil, errors.New("Activation of Activators is not a known type")
	}
}

func (Act *Activator) spawnEvaluators(f EvalFunc, X *EncInput, ch chan []int) {
	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		f(X, i, j, Act.poly, Act.box)
	}
}

//Evaluates a polynomial on the ciphertext
func (Act *Activator) ActivateBlocks(X *EncInput) {
	var f EvalFunc
	if !Act.isCheby {
		f = EvalPolyBlocks
	} else {
		f = EvalPolyBlocksVector
	}
	if Act.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				f(X, i, j, Act.poly, Act.box)
			}
		}
	} else if Act.poolSize > 1 {
		//bounded threading
		ch := make(chan []int)
		var wg sync.WaitGroup
		for i := 0; i < Act.poolSize; i++ {
			wg.Add(1)
			go func() {
				Act.spawnEvaluators(f, X, ch)
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
func EvalPolyBlocks(X *EncInput, i, j int, poly ActivationPoly, Box CkksBox) {
	eval := Box.Evaluator.ShallowCopy()
	ct, _ := eval.EvaluatePoly(X.Blocks[i][j], poly.polynomial, X.Blocks[i][j].Scale)
	eval.Add(ct, poly.term0VecEcd, ct)
	X.Blocks[i][j] = ct
}

//Evaluates a polynomial on the ciphertext using ckks.Evaluator.EvaluatePolyVector. This should be called via Activator.ActivateBlocks
func EvalPolyBlocksVector(X *EncInput, i, j int, poly ActivationPoly, Box CkksBox) {
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
