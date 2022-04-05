package main

import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

func main() {

	LDim := []int{128, 29}
	W0Dim := []int{29, 13}

	r := rand.New(rand.NewSource(0))

	L := make([][]float64, LDim[0])
	for i := range L {
		L[i] = make([]float64, LDim[1])

		for j := range L[i] {
			L[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < LDim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < LDim[1]; j++ {
			fmt.Printf("%7.4f, ", L[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

	W0 := make([][]float64, W0Dim[0])
	for i := range W0 {
		W0[i] = make([]float64, W0Dim[1])

		for j := range W0[i] {
			W0[i][j] = r.NormFloat64()
		}
	}

	fmt.Printf("[\n")
	for i := 0; i < W0Dim[0]; i++ {
		fmt.Printf("[")
		for j := 0; j < W0Dim[1]; j++ {
			fmt.Printf("%7.4f, ", W0[i][j])
		}
		fmt.Printf("],\n")
	}
	fmt.Printf("]\n")

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
	for i := 1; i < (len(W0)+1)>>1; i++ {
		rotations = append(rotations, 2*i*LDim[0])
	}

	rotations = append(rotations, len(L))
	rotations = append(rotations, len(W0))

	rotations = append(rotations, -len(W0)*len(L))

	rotations = append(rotations, params.RotationsForReplicateLog(LDim[0]*W0Dim[0], 3)...)
	rotations = append(rotations, params.RotationsForReplicateLog(LDim[0]*W0Dim[1], 3)...)
	rtks := kgen.GenRotationKeysForRotations(rotations, true, sk)

	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rtks})

	ctL := EncryptInput(params.MaxLevel(), L, params, ecd, enc)
	ctW0 := EncryptWeights(params.MaxLevel(), W0, len(L), params, ecd, enc)
	fmt.Println("//////////////////////////")

	now := time.Now()
	B := Dense(ctL, len(L), len(W0), len(W0[0]), ctW0, true, true, params, eval, ecd)
	// -> Activate
	fmt.Println("Done:", time.Since(now))

	fmt.Println("________________-")
	Lmat := mat.NewDense(LDim[0], LDim[1], plainUtils.Vectorize(L, true))
	W0mat := mat.NewDense(W0Dim[0], W0Dim[1], plainUtils.Vectorize(W0, true))

	resPt := dec.DecryptNew(B)
	resArray := ecd.DecodeSlots(resPt, 14)
	resReal := plainUtils.ComplexToReal(resArray)[:(len(L) * len(W0[0]))]
	var tmp mat.Dense
	tmp.Mul(Lmat, W0mat)

	resT := plainUtils.TransposeDense(&tmp)
	//for i := 0; i < plainUtils.NumRows(resT); i++ {
	//	fmt.Println(resT.RawRowView(i))
	//}
	fmt.Println("________________-")
	fmt.Println(plainUtils.Distance(plainUtils.RowFlatten(resT), resReal))

}

func FormatWeights(w [][]float64, leftdim int) (m [][]complex128) {

	scaling := complex(0.5, 0)

	m = make([][]complex128, (len(w)+1)/2)

	for i := 0; i < len(w)>>1; i++ {

		m[i] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {

			cReal := w[(i*2+0+j)%len(w)][j]
			cImag := w[(i*2+1+j)%len(w)][j]

			for k := 0; k < leftdim; k++ {
				fmt.Printf("Row %d col %d of m: real(%d,%d) imag(%d,%d)\n\n", i, j*leftdim+k, (i*2+0+j)%len(w), j, (i*2+1+j)%len(w), j)
				m[i][j*leftdim+k] = scaling * complex(cReal, -cImag) // 0.5 factor for imaginary part cleaning: (a+bi) + (a-bi) = 2a
			}
		}
	}

	if len(w)&1 == 1 {

		idx := len(m) - 1

		m[idx] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {
			cReal := w[(idx*2+j)%len(w)][j]
			for k := 0; k < leftdim; k++ {
				fmt.Printf("Row %d col %d of m: real(%d) imag(%d)\n\n", idx, j*leftdim+k, idx*2+j, -1)
				m[idx][j*leftdim+k] = scaling * complex(cReal, 0)
			}
		}
	}

	return
}

func EncryptWeights(level int, w [][]float64, leftdim int, params ckks.Parameters, ecd ckks.Encoder, enc ckks.Encryptor) (ctW []*ckks.Ciphertext) {
	wF := FormatWeights(w, leftdim)

	pt := ckks.NewPlaintext(params, level, params.QiFloat64(level))

	ctW = make([]*ckks.Ciphertext, len(wF))

	for i := range ctW {
		ecd.EncodeSlots(wF[i], pt, params.LogSlots())
		ctW[i] = enc.EncryptNew(pt)
	}

	return
}

func FormatInput(w [][]float64) (v []float64) {
	v = make([]float64, len(w)*len(w[0]))
	for i := 0; i < len(w[0]); i++ {
		for j := 0; j < len(w); j++ {
			v[i*len(w)+j] = w[j][i]
		}
	}

	return
}

func EncryptInput(level int, w [][]float64, params ckks.Parameters, ecd ckks.Encoder, enc ckks.Encryptor) (ctW *ckks.Ciphertext) {
	wF := FormatInput(w)
	pt := ckks.NewPlaintext(params, level, params.DefaultScale())
	ecd.EncodeSlots(wF, pt, params.LogSlots())
	return enc.EncryptNew(pt)
}

func Dense(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Ciphertext, prepack, cleanImag bool, params ckks.Parameters, eval ckks.Evaluator, ecd ckks.Encoder) (res *ckks.Ciphertext) {

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
