package cipherUtils

import "github.com/tuneinsight/lattigo/v3/ckks"

func Cipher2PMul(input *ckks.Ciphertext, dimIn, dimMid int, weights []*ckks.Plaintext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {
	/*
		Multiplies 1 ciphertexts to 1 plaintext, each representing a matrix
	*/
	params := Box.Params
	eval := Box.Evaluator

	var tmp *ckks.Ciphertext
	// Pack value for complex dot-product
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		tmp = eval.RotateNew(input, -dimMid*dimIn)
		eval.Add(tmp, input, tmp)

		img := eval.MultByiNew(tmp)
		eval.Rotate(img, dimIn, img)
		eval.Add(tmp, img, tmp)
		eval.Add(tmp, eval.RotateNew(tmp, -2*dimMid*dimIn), tmp)
	} else {
		tmp = input
	}
	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(tmp, weights[0])

	tmpRot := ckks.NewCiphertext(params, 1, tmp.Level(), tmp.Scale)
	eval.GetKeySwitcher().DecomposeNTT(tmp.Level(), params.PCount()-1, params.PCount(), tmp.Value[1], eval.GetKeySwitcher().PoolDecompQP)
	for i := 1; i < len(weights); i++ {
		eval.PermuteNTTHoisted(tmp.Level(), tmp.Value[0], tmp.Value[1], eval.GetKeySwitcher().PoolDecompQP, 2*dimIn*i, tmpRot.Value[0], tmpRot.Value[1])
		eval.MulAndAdd(tmpRot, weights[i], res)
	}

	// Rescale
	eval.Rescale(res, params.DefaultScale(), res)

	// Erases imaginary part
	if cleanImag {
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return
}

func Cipher2CMul(input *ckks.Ciphertext, dimIn, dimMid int, weights []*ckks.Ciphertext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {
	/*
		Multiplies 2 ciphertexts, each representing a matrix
	*/
	params := Box.Params
	eval := Box.Evaluator

	var tmp *ckks.Ciphertext
	// Pack value for complex dot-product
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		tmp = eval.RotateNew(input, -dimMid*dimIn)
		eval.Add(tmp, input, tmp)

		img := eval.MultByiNew(tmp)
		eval.Rotate(img, dimIn, img)
		eval.Add(tmp, img, tmp)
		eval.Add(tmp, eval.RotateNew(tmp, -2*dimMid*dimIn), tmp)
	} else {
		tmp = input
	}

	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(tmp, weights[0])

	tmpRot := ckks.NewCiphertext(params, 1, tmp.Level(), tmp.Scale)
	eval.GetKeySwitcher().DecomposeNTT(tmp.Level(), params.PCount()-1, params.PCount(), tmp.Value[1], eval.GetKeySwitcher().PoolDecompQP)
	for i := 1; i < len(weights); i++ {
		eval.PermuteNTTHoisted(tmp.Level(), tmp.Value[0], tmp.Value[1], eval.GetKeySwitcher().PoolDecompQP, 2*dimIn*i, tmpRot.Value[0], tmpRot.Value[1])
		eval.MulAndAdd(tmpRot, weights[i], res)
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
