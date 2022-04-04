package cipherUtils

import "github.com/tuneinsight/lattigo/v3/ckks"

func Cipher2PMul(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Plaintext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {
	/*
		Multiplies 1 ciphertexts to 1 plaintext, each representing a matrix
	*/
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
			eval.ReplicateLog(input, dimIn*dimMid, 3, input)
		} else {
			eval.ReplicateLog(input, dimIn*dimMid, 2, input)
		}
	}
	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(input, weights[0])

	inputRot := ckks.NewCiphertext(params, 1, input.Level(), input.Scale)

	eval.GetKeySwitcher().DecomposeNTT(input.Level(), params.PCount()-1, params.PCount(), input.Value[1], eval.GetKeySwitcher().PoolDecompQP)

	for i := 1; i < len(weights); i++ {

		eval.PermuteNTTHoisted(input.Level(), input.Value[0], input.Value[1], eval.GetKeySwitcher().PoolDecompQP, 2*dimIn*i, inputRot.Value[0], inputRot.Value[1])

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

func Cipher2CMul(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Ciphertext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {
	/*
		Multiplies 2 ciphertexts, each representing a matrix
	*/
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
			eval.ReplicateLog(input, dimIn*dimMid, 3, input)
		} else {
			eval.ReplicateLog(input, dimIn*dimMid, 2, input)
		}
	}

	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(input, weights[0])

	inputRot := ckks.NewCiphertext(params, 1, input.Level(), input.Scale)

	eval.GetKeySwitcher().DecomposeNTT(input.Level(), params.PCount()-1, params.PCount(), input.Value[1], eval.GetKeySwitcher().PoolDecompQP)

	for i := 1; i < len(weights); i++ {

		eval.PermuteNTTHoisted(input.Level(), input.Value[0], input.Value[1], eval.GetKeySwitcher().PoolDecompQP, 2*dimIn*i, inputRot.Value[0], inputRot.Value[1])

		eval.MulAndAdd(inputRot, weights[i], res)

	}

	// Rescale and relinearize
	eval.Relinearize(res, res)
	eval.Rescale(res, params.DefaultScale(), res)

	// Erases imaginary part
	if cleanImag {
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return
}
