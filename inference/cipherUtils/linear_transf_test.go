package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/stretchr/testify/assert"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
	"time"
)

var CNparamsLogN14, _ = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	LogN:         14,
	LogQ:         []int{45, 40, 40, 40, 40}, //Log(PQ) <= 438 for LogN 14
	LogP:         []int{60, 60},
	Sigma:        rlwe.DefaultSigma,
	LogSlots:     13,
	DefaultScale: float64(1 << 40),
})

//Assume W is square
func FormatWeightMap(W [][]float64, rowIn int, slots int) (map[int][]float64, error) {
	d := len(W)
	if d != len(W[0]) {
		return nil, errors.New("Non square")
	}
	if d*rowIn*2 > slots {
		return nil, errors.New("d * rowIn * 2 > slots")
	}
	nonZeroDiags := make(map[int][]float64) //rotation -> diag to be multiplied by
	for i := 0; i < d; i++ {
		isZero := true
		diag := make([]float64, d*rowIn)
		z := 0
		for j := 0; j < d; j++ {
			for k := 0; k < rowIn; k++ {
				diag[z] = W[(j+i)%d][(j)%d]
				z++
			}
			if diag[j] != 0 {
				isZero = false
			}
		}
		if !isZero {
			nonZeroDiags[rowIn*i] = plainUtils.ReplicateRealArray(diag, 2)
		}
	}
	return nonZeroDiags, nil
}

func MulWpt(input *ckks.Ciphertext, dimIn, dimMid, dimOut int, weights []*ckks.Plaintext, prepack, cleanImag bool, Box CkksBox) (res *ckks.Ciphertext) {

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

	// Lazy inner-product with hoisted rotations
	res = eval.MulNew(input, weights[0])

	inputRot := ckks.NewCiphertext(params, 1, input.Level(), input.Scale)

	eval.GetKeySwitcher().DecomposeNTT(input.Level(), params.PCount()-1, params.PCount(), input.Value[1], eval.GetKeySwitcher().BuffDecompQP)

	for i := 1; i < len(weights); i++ {

		eval.PermuteNTTHoisted(input.Level(), input.Value[0], input.Value[1], eval.GetKeySwitcher().BuffDecompQP, 2*dimIn*i, inputRot.Value[0], inputRot.Value[1])

		eval.MulAndAdd(inputRot, weights[i], res)

	}

	// Relin
	if res.Degree() > 1 {
		eval.Relinearize(res, res)
	}

	// rescales + erases imaginary part
	if cleanImag {
		eval.Rescale(res, params.DefaultScale(), res)
		eval.Add(res, eval.ConjugateNew(res), res)
	}

	return
}

func Test_LT(t *testing.T) {
	r := 64
	c := 64
	t.Run("Simple Mul", func(t *testing.T) {
		A := plainUtils.RandMatrix(r, c)
		W := plainUtils.RandMatrix(c, c)
		plainUtils.PrintDense(A)
		plainUtils.PrintDense(W)

		Box := NewBox(CNparamsLogN14)

		ctA := Box.Encryptor.EncryptNew(Box.Encoder.EncodeNew(plainUtils.RowFlatten(plainUtils.TransposeDense(A)), CNparamsLogN14.MaxLevel(), CNparamsLogN14.DefaultScale(), CNparamsLogN14.LogSlots()))
		diagW, err := FormatWeightMap(plainUtils.MatToArray(W), r, CNparamsLogN14.Slots())
		utils.ThrowErr(err)
		lt := ckks.GenLinearTransformBSGS(Box.Encoder, diagW, CNparamsLogN14.MaxLevel(), CNparamsLogN14.QiFloat64(CNparamsLogN14.MaxLevel()), 8, CNparamsLogN14.LogSlots())
		rotations := CNparamsLogN14.RotationsForReplicateLog(r*c, GetReplicaFactor(c, c))
		rotations = append(rotations, lt.Rotations()...)
		rotations = append(rotations, GenRotations(r, c, 1, []int{c}, []int{c}, []int{1}, []int{1}, CNparamsLogN14, nil)...)

		Box = BoxWithRotations(Box, rotations, false, bootstrapping.Parameters{})
		start := time.Now()
		Box.Evaluator.ReplicateLog(ctA, r*c, GetReplicaFactor(c, c), ctA)
		ctR := Box.Evaluator.LinearTransformNew(ctA, lt)[0]
		end1 := time.Since(start)

		ptW := EncodeWeights(CNparamsLogN14.MaxLevel(), plainUtils.MatToArray(W), r, Box)
		ctA = Box.Encryptor.EncryptNew(Box.Encoder.EncodeNew(plainUtils.RowFlatten(plainUtils.TransposeDense(A)), CNparamsLogN14.MaxLevel(), CNparamsLogN14.DefaultScale(), CNparamsLogN14.LogSlots()))
		start = time.Now()
		ctR2 := MulWpt(ctA, r, c, c, ptW, true, true, Box)
		end2 := time.Since(start)

		ReImg := Box.Encoder.Decode(Box.Decryptor.DecryptNew(ctR), CNparamsLogN14.LogSlots())[:(r * c)]
		Re := make([]float64, len(ReImg))
		for i := range ReImg {
			Re[i] = real(ReImg[i])
		}
		ReImg2 := Box.Encoder.Decode(Box.Decryptor.DecryptNew(ctR2), CNparamsLogN14.LogSlots())[:(r * c)]
		Re2 := make([]float64, len(ReImg2))
		for i := range ReImg2 {
			Re2[i] = real(ReImg2[i])
		}

		fmt.Println("Want")
		Rm := new(mat.Dense)
		Rm.Mul(A, W)
		R := plainUtils.RowFlatten(plainUtils.TransposeDense(Rm))
		fmt.Println(R)
		fmt.Println("Got -- lt with BSGS")
		fmt.Println(Re)
		fmt.Println("Got -- standard")
		fmt.Println(Re2)

		fmt.Println("Timings:")
		fmt.Println("LT: ", end1)
		fmt.Println("Standard: ", end2)

		for i := range R {
			assert.Less(t, math.Abs(R[i]-Re[i]), 1e-5, "BSGS fail")
		}

		for i := range R {
			assert.Less(t, math.Abs(R[i]-Re2[i]), 1e-5, "Std fail")
		}
	})

	t.Run("Mul-Act-Mul", func(t *testing.T) {
		A := plainUtils.RandMatrix(r, c)
		W1 := plainUtils.MulByConst(plainUtils.RandMatrix(c, c), 1e-3)
		W2 := plainUtils.MulByConst(plainUtils.RandMatrix(c, c), 1e-3)
		plainUtils.PrintDense(A)
		fmt.Println()
		plainUtils.PrintDense(W1)
		fmt.Println()
		plainUtils.PrintDense(W2)

		// Activation

		act := new(ckks.Polynomial)
		act.Coeffs = []complex128{complex(0, 0), complex(0, 0), complex(1, 0)} //f(x) = x^2
		null := new(ckks.Polynomial)
		null.Coeffs = []complex128{complex(0, 0), complex(0, 0), complex(0, 0)} //g(x) = 0
		slotsIndex := make(map[int][]int)

		idxF := make([]int, r*c)
		idxG := make([]int, CNparamsLogN14.Slots()-(r*c))
		for i := 0; i < CNparamsLogN14.Slots(); i++ {
			if i < (r * c) {
				idxF[i] = i // Index with all effective slots
			} else {
				idxG[i-(r*c)] = i // Index with all garbage slots
			}
		}

		slotsIndex[0] = idxF
		slotsIndex[1] = idxG

		Box := NewBox(CNparamsLogN14)

		//LT

		ctA := Box.Encryptor.EncryptNew(Box.Encoder.EncodeNew(plainUtils.RowFlatten(plainUtils.TransposeDense(A)), CNparamsLogN14.MaxLevel(), CNparamsLogN14.DefaultScale(), CNparamsLogN14.LogSlots()))
		diagW1, err := FormatWeightMap(plainUtils.MatToArray(W1), r, CNparamsLogN14.Slots())
		diagW2, err := FormatWeightMap(plainUtils.MatToArray(W2), r, CNparamsLogN14.Slots())
		utils.ThrowErr(err)
		lt1 := ckks.GenLinearTransformBSGS(Box.Encoder, diagW1, CNparamsLogN14.MaxLevel(), CNparamsLogN14.QiFloat64(CNparamsLogN14.MaxLevel()), 8, CNparamsLogN14.LogSlots())
		lt2 := ckks.GenLinearTransformBSGS(Box.Encoder, diagW2, CNparamsLogN14.MaxLevel()-2, CNparamsLogN14.QiFloat64(CNparamsLogN14.MaxLevel()-2), 8, CNparamsLogN14.LogSlots())
		rotations := CNparamsLogN14.RotationsForReplicateLog(r*c, GetReplicaFactor(c, c))
		rotations = append(rotations, lt1.Rotations()...)
		rotations = append(rotations, lt2.Rotations()...)
		rotations = append(rotations, GenRotations(r, c, 2, []int{c, c}, []int{c, c}, []int{1, 1}, []int{1, 1}, CNparamsLogN14, nil)...)

		Box = BoxWithRotations(Box, rotations, false, bootstrapping.Parameters{})

		start := time.Now()

		Box.Evaluator.ReplicateLog(ctA, r*c, GetReplicaFactor(c, c), ctA)
		ctR := Box.Evaluator.LinearTransformNew(ctA, lt1)[0]
		Box.Evaluator.Rescale(ctR, CNparamsLogN14.DefaultScale(), ctR)
		if ctR, err = Box.Evaluator.EvaluatePolyVector(ctR, []*ckks.Polynomial{act, null}, Box.Encoder, slotsIndex, ctR.Scale); err != nil {
			panic(err)
		}
		Box.Evaluator.ReplicateLog(ctR, r*c, GetReplicaFactor(c, c), ctR)
		Box.Evaluator.LinearTransform(ctR, lt2, []*ckks.Ciphertext{ctR})
		Box.Evaluator.Rescale(ctR, CNparamsLogN14.DefaultScale(), ctR)

		end := time.Since(start)

		ReImg := Box.Encoder.Decode(Box.Decryptor.DecryptNew(ctR), CNparamsLogN14.LogSlots())[:(r * c)]
		Re := make([]float64, len(ReImg))
		for i := range ReImg {
			Re[i] = real(ReImg[i])
		}

		fmt.Println("Want")
		Rm := new(mat.Dense)
		Rm.Mul(A, W1)
		Rm = plainUtils.ApplyFuncDense(func(v float64) float64 {
			return v * v
		}, Rm)
		R := plainUtils.RowFlatten(plainUtils.TransposeDense(Rm))
		Rm2 := new(mat.Dense)
		Rm2.Mul(Rm, W2)
		R = plainUtils.RowFlatten(plainUtils.TransposeDense(Rm2))
		fmt.Println(R)
		fmt.Println("Got -- lt with BSGS")
		fmt.Println(Re)

		fmt.Println("Timings:")
		fmt.Println("LT: ", end)

		for i := range R {
			assert.Less(t, math.Abs(R[i]-Re[i]), 1e-5, "BSGS fail")
		}
		fmt.Println("Done... Consumed levels:", CNparamsLogN14.MaxLevel()-ctR.Level())

		//Standard
		ptW := EncodeWeights(CNparamsLogN14.MaxLevel(), plainUtils.MatToArray(W1), r, Box)
		ptW2 := EncodeWeights(CNparamsLogN14.MaxLevel()-3, plainUtils.MatToArray(W2), r, Box)
		ctA = Box.Encryptor.EncryptNew(Box.Encoder.EncodeNew(plainUtils.RowFlatten(plainUtils.TransposeDense(A)), CNparamsLogN14.MaxLevel(), CNparamsLogN14.DefaultScale(), CNparamsLogN14.LogSlots()))
		start = time.Now()
		ctR = MulWpt(ctA, r, c, c, ptW, true, true, Box)
		ctR, _ = Box.Evaluator.EvaluatePoly(ctR, act, ctR.Scale)
		ctR = MulWpt(ctR, r, c, c, ptW2, true, true, Box)
		end2 := time.Since(start)

		fmt.Println("Done... Consumed levels:", CNparamsLogN14.MaxLevel()-ctR.Level())

		ReImg = Box.Encoder.Decode(Box.Decryptor.DecryptNew(ctR), CNparamsLogN14.LogSlots())[:(r * c)]
		Re = make([]float64, len(ReImg))
		for i := range ReImg {
			Re[i] = real(ReImg[i])
		}

		fmt.Println("Want")
		fmt.Println(R)
		fmt.Println("Got -- Std")
		fmt.Println(Re)

		fmt.Println("Timings:")
		fmt.Println("STD: ", end2)
	})
}
