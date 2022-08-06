package cipherUtils

import (
	"crypto/rand"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ring"
	"math"
	"math/big"
)

//Generates a random mask which can be encoded and added as a ckks plaintext
func secureRandMask(n int, scale float64, lvl0 float64) []float64 {
	m := make([]float64, n)
	bitsLvl0 := math.Floor(math.Log(lvl0))
	bitsScale := math.Floor(math.Log(scale))
	for i := range m {
		r, _ := rand.Int(rand.Reader, big.NewInt(int64(math.Pow(2, bitsLvl0-bitsScale))))
		m[i] = float64(r.Uint64())
	}
	return m
}

//Deprecated
//Masks input by adding a random encoded mask as a random ckks plaintext
func MaskInputEcdMask(Xenc *EncInput, Box CkksBox) *PlainInput {
	mask := make([][]*ckks.Plaintext, len(Xenc.Blocks))
	for i := range Xenc.Blocks {
		mask[i] = make([]*ckks.Plaintext, len(Xenc.Blocks[i]))
		for j := range Xenc.Blocks[i] {
			m := secureRandMask(Xenc.InnerRows*Xenc.InnerCols, Box.Params.DefaultScale(), Box.Params.QiFloat64(0))
			mask[i][j] = Box.Encoder.EncodeNew(m, Xenc.Blocks[i][j].Level(), Box.Params.DefaultScale(), Box.Params.LogSlots())
			Box.Evaluator.Add(Xenc.Blocks[i][j], mask[i][j], Xenc.Blocks[i][j])
		}
	}
	return &PlainInput{
		Blocks:    mask,
		RowP:      Xenc.RowP,
		ColP:      Xenc.ColP,
		InnerRows: Xenc.InnerRows,
		InnerCols: Xenc.InnerCols,
	}
}

//generate a random mask for lambda bit security. Returns mask (for unmasking) and -mask (for masking)
func generateMask(params ckks.Parameters, level int, scale float64, lambda int) (*ckks.Plaintext, *ckks.Plaintext) {
	ringQ := params.RingQ()

	logBound := lambda //+ int(math.Ceil(math.Log2(scale)))

	levelQ := level

	// Get the upperbound on the norm
	// Ensures that bound >= 2^{128+logbound}
	bound := ring.NewUint(1)
	bound.Lsh(bound, uint(logBound))

	boundMax := ring.NewUint(ringQ.Modulus[0])
	for i := 1; i < levelQ+1; i++ {
		boundMax.Mul(boundMax, ring.NewUint(ringQ.Modulus[i]))
	}

	var sign int

	sign = bound.Cmp(boundMax)

	if sign == 1 || bound.Cmp(boundMax) == 1 {
		panic("ciphertext level is not large enough for refresh correctness")
	}

	boundHalf := new(big.Int).Rsh(bound, 1)

	dslots := 1 << params.LogSlots()
	//if ringQ.Type() == ring.Standard {
	//	dslots *= 2
	//}

	// Generate the mask in Z[Y] for Y = X^{N/(2*slots)}
	maskBigint := make([]*big.Int, dslots)
	maskBigintNeg := make([]*big.Int, dslots)
	for i := 0; i < dslots; i++ {
		maskBigint[i] = ring.RandInt(bound)
		sign = maskBigint[i].Cmp(boundHalf)
		if sign == 1 || sign == 0 {
			maskBigint[i].Sub(maskBigint[i], bound)
		}
		maskBigintNeg[i] = new(big.Int)
		maskBigintNeg[i] = maskBigintNeg[i].Neg(maskBigint[i])
	}
	maskPoly := ringQ.NewPoly()
	ringQ.SetCoefficientsBigintLvl(levelQ, maskBigint[:dslots], maskPoly)
	ckks.NttAndMontgomeryLvl(levelQ, params.LogSlots(), ringQ, false, maskPoly)
	maskPt := ckks.NewPlaintext(params, level, scale)
	maskPt.Value = maskPoly
	maskPt.Value.IsNTT = true

	maskPoly = ringQ.NewPoly()
	ringQ.SetCoefficientsBigintLvl(levelQ, maskBigintNeg[:dslots], maskPoly)
	ckks.NttAndMontgomeryLvl(levelQ, params.LogSlots(), ringQ, false, maskPoly)
	maskPtNeg := ckks.NewPlaintext(params, level, scale)
	maskPtNeg.Value = maskPoly
	maskPtNeg.Value.IsNTT = true

	return maskPt, maskPtNeg
}

//Masks ct with 128 bit security mask: ct = ct - mask. Ensures indistinguishabilty
func Mask(ct *ckks.Ciphertext, Box CkksBox) *ckks.Plaintext {
	mask, maskNeg := generateMask(Box.Params, ct.Level(), ct.Scale, 128)
	Box.Evaluator.Add(ct, maskNeg, ct)
	return mask
}

//Removes mask from pt: pt = pt + mask = (msg - mask) + mask
func UnMask(pt, mask *ckks.Plaintext, Box CkksBox) {
	kgen := ckks.NewKeyGenerator(Box.Params)
	skEph := kgen.GenSecretKey()
	ct := Box.Encryptor.WithKey(skEph).EncryptNew(pt)
	Box.Evaluator.Add(ct, mask, ct)
	Box.Decryptor.WithKey(skEph).Decrypt(ct, pt)
}

//mask input with lambda bit security
func MaskInput(Xenc *EncInput, Box CkksBox, lambda int) *PlainInput {
	// Generate the mask in Z[Y] for Y = X^{N/(2*slots)}
	maskBlocks := make([][]*ckks.Plaintext, Xenc.RowP)
	for i := 0; i < Xenc.RowP; i++ {
		for j := 0; j < Xenc.ColP; j++ {
			maskBlocks[i][j] = Mask(Xenc.Blocks[i][j], Box)
		}
	}
	return &PlainInput{
		Blocks:    maskBlocks,
		RowP:      Xenc.RowP,
		ColP:      Xenc.ColP,
		InnerRows: Xenc.InnerRows,
		InnerCols: Xenc.InnerCols,
	}
}

//removes mask from MaskInput
func UnmaskInput(Xmask, mask *PlainInput, Box CkksBox) {
	for i := 0; i < Xmask.RowP; i++ {
		for j := 0; j < Xmask.ColP; j++ {
			UnMask(Xmask.Blocks[i][j], mask.Blocks[i][j], Box)
		}
	}
}
