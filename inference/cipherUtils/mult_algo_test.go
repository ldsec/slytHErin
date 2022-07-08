package cipherUtils

import (
	"fmt"
	pU "github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"gonum.org/v1/gonum/mat"
	"testing"
)

/********************************************
EXPERIMENTS ON MULTIPLICATION ALGORITHM
*********************************************/

func TestMultiplicationAlgo(t *testing.T) {
	rowx := 87
	colx := 87
	roww := 87
	colw := 87
	//X := pU.RandMatrix(rowx, colx)
	//W := pU.RandMatrix(roww, colw)
	Xv := make([][]float64, rowx)
	for i := 0; i < rowx; i++ {
		Xv[i] = make([]float64, colx)
		for j := 0; j < colx; j++ {
			Xv[i][j] = 1.0 + float64(i*colx+j)*1e-0
		}
	}
	X := pU.NewDense(Xv)
	Wv := make([][]float64, roww)
	for i := 0; i < roww; i++ {
		Wv[i] = make([]float64, colw)
		for j := 0; j < colw; j++ {
			Wv[i][j] = 1.0 + float64(i*colw+j)*1e-0
			fmt.Println(Wv[i][j])
		}
		fmt.Println()
	}
	W := pU.NewDense(Wv)

	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)

	Box := NewBox(params)
	Box = BoxWithEvaluators(Box, bootstrapping.Parameters{}, false, pU.NumRows(X), pU.NumCols(X), 1, []int{pU.NumRows(W)}, []int{pU.NumCols(W)})

	t.Run("Test/C2P", func(t *testing.T) {
		//does not work when input cols < rows, or when dimOut > dimMid
		tmp1 := make([]complex128, params.Slots())
		input := EncryptInput(params.MaxLevel(), params.DefaultScale(), pU.MatToArray(X), Box)
		Wenc := EncodeWeights(params.MaxLevel()-1, pU.MatToArray(W), pU.NumRows(X), Box)
		weights := make([]ckks.Operand, len(Wenc))
		for i := range weights {
			weights[i] = Wenc[i]
		}
		eval := Box.Evaluator
		dimIn := pU.NumRows(X)
		dimMid := pU.NumCols(X)

		img := eval.MultByiNew(input)
		eval.Rotate(img, dimIn, img)

		eval.Add(input, img, input)

		mask := make([]float64, params.Slots())
		for i := range mask {
			if i >= params.Slots()-dimIn {
				mask[i] = 1
			}
		}
		maskEcd := Box.Encoder.EncodeNew(mask, img.Level(), params.QiFloat64(img.Level()), params.LogSlots())
		eval.Mul(img, maskEcd, img)                      //keeps only first col in last slots
		eval.Rotate(img, -dimIn-(dimIn*(dimMid-1)), img) //puts first col in last col
		eval.Rescale(img, params.DefaultScale(), img)
		tmp1 = Box.Encoder.Decode(Box.Decryptor.DecryptNew(img), params.LogSlots())
		fmt.Println(tmp1[:dimMid*dimIn])

		// Lazy inner-product with hoisted rotations
		rotations := make([]int, len(weights)-1)
		for i := 1; i < len(weights); i++ {
			rotations[i-1] = 2 * dimIn * i
		}
		imgRot := eval.RotateHoistedNew(img, rotations) //rotated versions of first col

		tmp := eval.AddNew(input, img)
		tmp1 = Box.Encoder.Decode(Box.Decryptor.DecryptNew(tmp), params.LogSlots())
		fmt.Println(0)
		fmt.Println(tmp1[:dimMid*dimIn])
		tmp1 = Box.Encoder.Decode(weights[0].(*ckks.Plaintext), params.LogSlots())
		fmt.Println(0)
		//fmt.Println(tmp1[:dimMid*dimIn*pU.NumCols(W)])
		res := eval.MulNew(tmp, weights[0])

		LtsInput := make([]ckks.LinearTransform, len(weights)-1)

		for i := 1; i < len(weights); i++ {
			LtsInput[i-1] = GenSubVectorRotationMatrix(params, input.Level(), params.QiFloat64(input.Level()), dimIn*dimMid, 2*dimIn*i, params.LogSlots(), Box.Encoder)
		}
		if len(LtsInput) > 0 {
			inputRots := eval.LinearTransformNew(input, LtsInput)
			tmp1 = Box.Encoder.Decode(Box.Decryptor.DecryptNew(inputRots[0]), params.LogSlots())
			fmt.Println(tmp1[:dimMid*dimIn])

			for i := 1; i < len(weights); i++ {
				eval.Rescale(inputRots[i-1], params.DefaultScale(), inputRots[i-1])
				fmt.Println(i)
				tmp := eval.AddNew(inputRots[i-1], imgRot[2*dimIn*i])
				tmp1 = Box.Encoder.Decode(Box.Decryptor.DecryptNew(tmp), params.LogSlots())
				fmt.Println(tmp1[:dimMid*dimIn])
				eval.MulAndAdd(tmp, weights[i], res)
			}
		}

		// Rescale
		if res.Degree() > 1 {
			eval.Relinearize(res, res)
		}

		// rescales + erases imaginary part
		eval.Rescale(res, params.DefaultScale(), res)
		fmt.Println("Level drop res:", params.MaxLevel()-res.Level())
		fmt.Println("scale diff:", params.DefaultScale()-res.Scale)
		eval.Add(res, eval.ConjugateNew(res), res)

		var resPlain mat.Dense
		resPlain.Mul(X, W)

		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(res, valuesWant, 1, Box)
	})

	t.Run("Test/C2C", func(t *testing.T) {
		Xenc := EncryptInput(params.MaxLevel(), params.DefaultScale(), pU.MatToArray(X), Box)
		Wenc := EncryptWeights(params.MaxLevel(), pU.MatToArray(W), pU.NumRows(X), Box)
		ops := make([]ckks.Operand, len(Wenc))
		for i := range Wenc {
			ops[i] = Wenc[i]
		}
		Renc := DiagMul(Xenc, pU.NumRows(X), pU.NumCols(X), pU.NumCols(W), ops, true, true, Box)
		var resPlain mat.Dense
		resPlain.Mul(X, W)

		//we need to tranpose the plaintext result according to the diagonalized multiplication algo
		valuesWant := pU.RealToComplex(pU.Vectorize(pU.MatToArray(&resPlain), false))
		PrintDebug(Renc, valuesWant, 0.001, Box)
	})

}
