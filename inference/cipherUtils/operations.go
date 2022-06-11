package cipherUtils

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
)

/*
	Operations between encrypted matrices of data
*/
// |
// | version with optimized dimentions
// v

//Multiplies 1 ciphertexts to 1 plaintext, each representing a matrix
func Cipher2PMul(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Plaintext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {

	params := Box.Params
	eval := Box.Evaluator
	// Pack value for complex dot-product
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		img := eval.MultByiNew(input)
		eval.Rotate(img, dimIn, img)
		eval.Add(input, img, input)
		if dimMid < dimOut {
			//3 x space needed
			eval.ReplicateLog(input, dimIn*dimMid, 3, input)
		} else {
			// 2 x space needed
			eval.ReplicateLog(input, dimIn*dimMid, 2, input)
		}
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
	eval.Rescale(res, params.DefaultScale(), res)

	// Erases imaginary part
	if cleanImag {
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return

}

//Multiplies 2 ciphertexts, each representing a matrix
func Cipher2CMul(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Ciphertext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {

	params := Box.Params
	eval := Box.Evaluator

	// Pack value for complex dot-product
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		img := eval.MultByiNew(input)
		eval.Rotate(img, dimIn, img)
		eval.Add(input, img, input)
		if dimMid < dimOut {
			//3 x space needed
			eval.ReplicateLog(input, dimIn*dimMid, 3, input)
		} else {
			// 2 x space needed
			eval.ReplicateLog(input, dimIn*dimMid, 2, input)
		}
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
	eval.Relinearize(res, res)
	eval.Rescale(res, params.DefaultScale(), res)

	// Erases imaginary part
	if cleanImag {
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return
}

func Cipher2CMul_Debug(input *ckks.Ciphertext, inputPlain []complex128, dimIn, dimMid, dimOut int, weights []*ckks.Ciphertext, weightsPlain [][]complex128, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {

	params := Box.Params
	eval := Box.Evaluator

	// Pack value for complex dot-product
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		img := eval.MultByiNew(input)
		eval.Rotate(img, dimIn, img)
		eval.Add(input, img, input)

		imgPlain := plainUtils.MulByi(plainUtils.ComplexToReal(inputPlain))
		imgPlain = plainUtils.RotateComplexArray(imgPlain, dimIn)
		for i := range imgPlain {
			inputPlain[i] = inputPlain[i] + imgPlain[i]
		}
		fmt.Println("before replication")
		PrintDebug(input, inputPlain, Box)
		if dimMid < dimOut {
			//3 x space needed
			replicaFactor := int(math.Floor(float64(dimOut/dimMid))) + 1
			eval.ReplicateLog(input, dimIn*dimMid, replicaFactor, input)
			inputPlain = plainUtils.ReplicateComplexArray(inputPlain[:dimIn*dimMid], replicaFactor)
		} else {
			// 2 x space needed
			eval.ReplicateLog(input, dimIn*dimMid, 2, input)
			inputPlain = plainUtils.ReplicateComplexArray(inputPlain[:dimIn*dimMid], 2)
		}
		fmt.Println("Done prepacking")
		PrintDebug(input, inputPlain, Box)
	}

	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(input, weights[0])
	resPlain := make([]complex128, len(inputPlain))
	for i := range weightsPlain[0] {
		resPlain[i] = inputPlain[i] * weightsPlain[0][i]
	}

	inputRot := ckks.NewCiphertext(params, 1, input.Level(), input.Scale)

	eval.GetKeySwitcher().DecomposeNTT(input.Level(), params.PCount()-1, params.PCount(), input.Value[1], eval.GetKeySwitcher().BuffDecompQP)

	for i := 1; i < len(weights); i++ {

		eval.PermuteNTTHoisted(input.Level(), input.Value[0], input.Value[1], eval.GetKeySwitcher().BuffDecompQP, 2*dimIn*i, inputRot.Value[0], inputRot.Value[1])

		eval.MulAndAdd(inputRot, weights[i], res)

	}

	// Rescale
	eval.Relinearize(res, res)
	eval.Rescale(res, params.DefaultScale(), res)
	fmt.Println("result of diag mul")
	PrintDebug(res, resPlain, Box)
	// Erases imaginary part
	if cleanImag {
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return
}
