package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"sync"
)

//Handles an activation layer with a polynomial function
type Activator struct {
	poly        *ckks.Polynomial
	term0       complex128 //constant term of the polynomial
	term0VecEcd *ckks.Plaintext
	Box         CkksBox //pool of evaluators to be used
	isCheby     bool
}

//Creates a new Activator. Takes lavel, scale, as well as the blocks and sub-matrices dimentions, of the
//output of the previous linear layer
func NewActivator(act interface{}, level int, scale float64, innerRows, innerCols, rowP, colP int, Box CkksBox) (*Activator, error) {
	Act := new(Activator)
	poly := new(ckks.Polynomial)
	switch act.(type) {
	case *utils.MinMaxPolyApprox:
		Act.term0 = act.(*utils.MinMaxPolyApprox).Poly.Coeffs[0]
		poly = act.(*utils.MinMaxPolyApprox).Poly
		Act.isCheby = false
		term0vec := make([]complex128, innerRows*innerCols)
		for i := range term0vec {
			term0vec[i] = Act.term0
		}
		if poly.Depth() > level {
			return nil, errors.New("Not enough levels for poly depth")
		}
		Act.term0VecEcd = ckks.NewPlaintext(Box.Params, level-poly.Depth(), scale)
		Box.Encoder.EncodeSlots(term0vec, Act.term0VecEcd, Box.Params.LogSlots())

		//copy the polynomial with the 0 const term without affecting the original poly
		polyCp := new(ckks.Polynomial)
		*polyCp = *poly
		polyCp.Coeffs = make([]complex128, len(poly.Coeffs))
		copy(polyCp.Coeffs, poly.Coeffs)
		polyCp.Coeffs[0] = complex(0, 0)
		Act.poly = polyCp

		Act.Box = Box
		return Act, nil

	case *utils.ChebyPolyApprox:
		Act.isCheby = true
		Act.poly = act.(*utils.ChebyPolyApprox).Poly
		Act.Box = Box
		return Act, nil
	default:
		return nil, errors.New("Activation of Activators is not a known type")
	}
}

func (act *Activator) ActivateBlocks(X *EncInput) {
	if !act.isCheby {
		var wg sync.WaitGroup
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				wg.Add(1)
				go func(eval ckks.Evaluator, poly *ckks.Polynomial, term0VecEcd *ckks.Plaintext, i, j int) {
					defer wg.Done()
					ct, _ := eval.EvaluatePoly(X.Blocks[i][j], poly, X.Blocks[i][j].Scale)
					eval.Add(ct, act.term0VecEcd, ct)
					X.Blocks[i][j] = ct
				}(act.Box.Evaluator.ShallowCopy(), act.poly, act.term0VecEcd, i, j)
			}
		}
		wg.Wait()
	} else {
		EvalPolyBlocks(X, act.poly, act.Box)
	}
}
