package main

/*



import (
	//"fmt"
	//"github.com/ldsec/dnn-inference/inference/plainUtils"
	//"github.com/tuneinsight/lattigo/v3/ckks"
	//"github.com/tuneinsight/lattigo/v3/rlwe"
	//"time"
	//"math"
	"fmt"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"time"
)

func main() {

		LDim := []int{3, 3}
		W0Dim := []int{3, 3}
		W1Dim := []int{3, 3}

		//r := rand.New(rand.NewSource(0))

		L := make([][]float64, LDim[0])
		for i := range L {
			L[i] = make([]float64, LDim[1])

			for j := range L[i] {
				L[i][j] = float64(i*LDim[0]+j)
			}
		}
		fmt.Println("L:", L)
		W0 := make([][]float64, W0Dim[0])
		for i := range W0 {
			W0[i] = make([]float64, W0Dim[1])

			for j := range W0[i] {
				W0[i][j]=float64(i*W0Dim[0]+j)
			}
		}
		fmt.Println("W0:", W0)

		W1 := make([][]float64, W1Dim[0])
		for i := range W1 {
			W1[i] = make([]float64, W1Dim[1])

			for j := range W1[i] {
				W1[i][j]=float64(i*W1Dim[0]+j)
			}
		}
		fmt.Println("W1", W1)

		// Schemes parameters are created from scratch
		params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:         15,
			LogQ:         []int{60, 60, 60, 40, 40},
			LogP:         []int{61, 61},
			Sigma:        rlwe.DefaultSigma,
			LogSlots:     14,
			DefaultScale: float64(1 << 40),
		})
		if err != nil {
			panic(err)
		}

		kgen := ckks.NewKeyGenerator(params)
		sk := kgen.GenSecretKey()
		rlk := kgen.GenRelinearizationKey(sk, 2)

		rotations := []int{}
		for i := 1; i < len(W0); i++ {
			rotations = append(rotations, 2*i*len(W0))
		}

		for i := 1; i < len(W1); i++ {
			rotations = append(rotations, 2*i*len(W1))
		}

		rotations = append(rotations, len(L))
		rotations = append(rotations, len(W0))
		rotations = append(rotations, len(W1))
		rotations = append(rotations, -len(W0)*len(L))
		rotations = append(rotations, -2*len(W0)*len(L))
		rotations = append(rotations, -len(W1)*len(L))
		rotations = append(rotations, -2*len(W1)*len(L))

		rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)

		enc := ckks.NewEncryptor(params, sk)
		dec := ckks.NewDecryptor(params, sk)
		ecd := ckks.NewEncoder(params)
		eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})

		ctW0 := EncryptWeights(params.MaxLevel(), W0, len(L), params, ecd, enc)
		ctW1 := EncryptWeights(params.MaxLevel(), W1, len(L), params, ecd, enc)
		ctA := EncryptInput(params.MaxLevel(), L, params, ecd, enc)

		now := time.Now()
		B := Dense(ctA, len(L), len(W0), len(W0[0]), ctW0, true, true, params, eval, ecd)
		// -> Activate
		fmt.Println("Done:", time.Since(now))

		now = time.Now()
		C := Dense(B, len(L), len(W1), len(W1[0]), ctW1, true, true, params, eval, ecd)
		// -> Activate
		fmt.Println("Done:", time.Since(now))
		resPt := dec.DecryptNew(C)
		resArray := ecd.DecodeSlots(resPt, 14)

	fmt.Println(resArray[0:9])
	/*
	X := make([][]float64, 3)
	for i := range X {
		X[i] = make([]float64, 3)
		for j := range X[i] {
			X[i][j] = float64(i*3 + j)
		}
	}
	fmt.Println(plainUtils.Vectorize(X))
	P := plainUtils.GenPaddingMatrixes(3,1)

}

func FormatWeights(w [][]float64, leftdim int) (m [][]complex128) {
	m = make([][]complex128, (len(w)+1)/2)

	for i := 0; i < len(w)>>1; i++ {

		m[i] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {

			cReal := w[(i*2+0+j)%len(w)][j]
			fmt.Println("cReal", cReal)
			cImag := w[(i*2+1+j)%len(w)][j]
			fmt.Println("cImag", cImag)

			for k := 0; k < leftdim; k++ {
				fmt.Printf("m value at place %d,%d = 0.5(%f, %f.i)\n", i, j*leftdim+k, cReal, -cImag)
				m[i][j*leftdim+k] = 0.5 * complex(cReal, -cImag) // 0.5 factor for imaginary part cleaning: (a+bi) + (a-bi) = 2a
			}
		}
	}

	if len(w)&1 == 1 {

		idx := len(m) - 1

		m[idx] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {
			cReal := w[(idx*2+j)%len(w)][j]
			for k := 0; k < leftdim; k++ {
				fmt.Printf("m value at place %d,%d = 0.5(%f, %f.i)\n", idx, j*leftdim+k, cReal, 0.0)
				m[idx][j*leftdim+k] = 0.5 * complex(cReal, 0)
			}
		}
	}

	return
}

func EncryptWeights(level int, w [][]float64, leftdim int, params ckks.Parameters, ecd ckks.Encoder, enc ckks.Encryptor) (ctW []*ckks.Ciphertext) {
	wF := FormatWeights(w, leftdim)
	fmt.Println("W:", w)
	fmt.Println("Wf:", wF)
	//fmt.Println(len(w), len(w[0]))
	//fmt.Println(len(wF), len(wF[0]))
	pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))

	ctW = make([]*ckks.Ciphertext, len(wF))

	for i := range ctW {
		//this puts the values in the plaintext in a way such that homomorphic elem-wise mult
		//is preserved
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ctW[i] = enc.EncryptNew(pt)
	}

	return
}

func FormatInput(w [][]float64) (v []float64) {
	//transpose matrix and padded with nxm 0s (w is nxm) --> len is 2xnxm
	v = make([]float64, len(w)*len(w[0])*2)

	for i := 0; i < len(w[0]); i++ {
		for j := 0; j < len(w); j++ {
			v[i*len(w)+j] = w[j][i]
		}
	}
	fmt.Println("Input:", w)
	fmt.Println("Formatted Input:", v)
	return
}

func EncryptInput(level int, w [][]float64, params ckks.Parameters, ecd ckks.Encoder, enc ckks.Encryptor) (ctW *ckks.Ciphertext) {
	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, params.DefaultScale())
	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return enc.EncryptNew(pt)
}

func Dense(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Ciphertext, prepack, cleanImag bool, params ckks.Parameters, eval ckks.Evaluator, ecd ckks.Encoder) (res *ckks.Ciphertext) {

	var tmp *ckks.Ciphertext
	// Pack input value for complex dot-product
	//	weight     input
	// formatted   packed
	// (a - bi) * (c + di) = (ac + bd) + i*garbage
	// This repack can be done during the refresh to save noise and reduce the number of slots used.
	if prepack {
		tmp = eval.RotateNew(input, -dimMid*dimIn) // e.g if we are multiplying a 2*2 x 2*2 this is rot -4
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
		eval.PermuteNTTHoisted(tmp.Level(), tmp.Value[0], tmp.Value[1], eval.GetKeySwitcher().PoolDecompQP, 2*dimMid*i, tmpRot.Value[0], tmpRot.Value[1])
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
*/
