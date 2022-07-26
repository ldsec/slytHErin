package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/ring"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/big"
)

// Interface for generic block matrix. In can be a plaintext or ciphertext matrix
type BlocksOperand interface {
	GetBlock(i, j int) []ckks.Operand
	GetPartitions() (int, int)
	GetInnerDims() (int, int)
	Level() int
	Scale() float64
}

//Encrypted block matrix, to be used for input or bias layer
type EncInput struct {
	BlocksOperand
	Blocks     [][]*ckks.Ciphertext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP int                  //num of partitions
	InnerRows  int                  //rows of sub-matrix
	InnerCols  int
}

//Plaintext block matrix, to be used for input or bias layer
type PlainInput struct {
	BlocksOperand
	Blocks     [][]*ckks.Plaintext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP int                 //num of partitions
	InnerRows  int                 //rows of sub-matrix
	InnerCols  int
}

//Encrypted matrix in diagonal form
type EncDiagMat struct {
	BlocksOperand
	//store an encrypted weight matrix in diagonal form
	Diags []*ckks.Ciphertext //enc diagonals
}

//Encrypted block matrix, weight of dense or convolutional layer
type EncWeightDiag struct {
	BlocksOperand
	Blocks     [][]*EncDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP int
	LeftDim    int //rows of left matrix
	NumDiags   int
	InnerRows  int //rows of matrix
	InnerCols  int
}

//Plaintext matrix in diagonal form
type PlainDiagMat struct {
	//store a plaintext weight matrix in diagonal form
	Diags []*ckks.Plaintext //diagonals
}

//Plaintext block matrix, weight of dense or convolutional layer
type PlainWeightDiag struct {
	Blocks     [][]*PlainDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP int
	LeftDim    int //rows of left matrix
	NumDiags   int
	InnerRows  int //rows of matrix
	InnerCols  int
}

func NewPlainInput(Xm *mat.Dense, rowP, colP int, level int, scale float64, Box CkksBox) (*PlainInput, error) {
	Xb, err := plainUtils.PartitionMatrix(Xm, rowP, colP)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	XPlain := new(PlainInput)
	XPlain.RowP = rowP
	XPlain.ColP = colP
	XPlain.InnerRows = Xb.InnerRows
	XPlain.InnerCols = Xb.InnerCols
	XPlain.Blocks = make([][]*ckks.Plaintext, rowP)
	for i := 0; i < rowP; i++ {
		XPlain.Blocks[i] = make([]*ckks.Plaintext, colP)
		for j := 0; j < colP; j++ {
			XPlain.Blocks[i][j] = EncodeInput(level, scale, plainUtils.MatToArray(Xb.Blocks[i][j]), Box)
		}
	}
	return XPlain, nil
}

func NewEncInput(Xm *mat.Dense, rowP, colP int, level int, scale float64, Box CkksBox) (*EncInput, error) {
	Xb, err := plainUtils.PartitionMatrix(Xm, rowP, colP)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	XEnc := new(EncInput)
	XEnc.RowP = rowP
	XEnc.ColP = colP
	XEnc.InnerRows = Xb.InnerRows
	XEnc.InnerCols = Xb.InnerCols
	if float64(XEnc.InnerRows*XEnc.InnerCols*2) > math.Pow(2, float64(Box.Params.LogSlots())) {
		utils.ThrowErr(errors.New(fmt.Sprintf("Input submatrixes elements must be less than 2^(LogSlots-1): had %d*%d", XEnc.InnerRows, XEnc.InnerCols)))
	}
	XEnc.Blocks = make([][]*ckks.Ciphertext, rowP)
	for i := 0; i < rowP; i++ {
		XEnc.Blocks[i] = make([]*ckks.Ciphertext, colP)
		for j := 0; j < colP; j++ {
			XEnc.Blocks[i][j] = EncryptInput(level, scale, plainUtils.MatToArray(Xb.Blocks[i][j]), Box)
		}
	}
	return XEnc, nil
}

func (X *EncInput) GetBlock(i, j int) []ckks.Operand {
	return []ckks.Operand{X.Blocks[i][j]}
}

func (X *EncInput) GetPartitions() (int, int) {
	return X.RowP, X.ColP
}

func (X *EncInput) GetInnerDims() (int, int) {
	return X.InnerRows, X.InnerCols
}

func (X *EncInput) Level() int {
	return X.GetBlock(0, 0)[0].Level()
}

func (X *EncInput) Scale() float64 {
	return X.GetBlock(0, 0)[0].ScalingFactor()
}

func (X *PlainInput) GetBlock(i, j int) []ckks.Operand {
	return []ckks.Operand{X.Blocks[i][j]}
}

func (X *PlainInput) GetPartitions() (int, int) {
	return X.RowP, X.ColP
}

func (X *PlainInput) GetInnerDims() (int, int) {
	return X.InnerRows, X.InnerCols
}

func (X *PlainInput) Level() int {
	return X.GetBlock(0, 0)[0].Level()
}

func (X *PlainInput) Scale() float64 {
	return X.GetBlock(0, 0)[0].ScalingFactor()
}

//	Given a block input matrix, decrypts and returns the underlying original matrix
//	The sub-matrices are also transposed (remember that they are in form flatten(A.T))
func DecInput(XEnc *EncInput, Box CkksBox) [][]float64 {

	Xb := new(plainUtils.BMatrix)
	Xb.RowP = XEnc.RowP
	Xb.ColP = XEnc.ColP
	Xb.InnerRows = XEnc.InnerRows
	Xb.InnerCols = XEnc.InnerCols
	Xb.Blocks = make([][]*mat.Dense, Xb.RowP)
	for i := 0; i < XEnc.RowP; i++ {
		Xb.Blocks[i] = make([]*mat.Dense, Xb.ColP)
		for j := 0; j < XEnc.ColP; j++ {
			pt := Box.Decryptor.DecryptNew(XEnc.Blocks[i][j])
			ptArray := Box.Encoder.DecodeSlots(pt, Box.Params.LogSlots())
			//this is flatten(x.T)
			resReal := plainUtils.ComplexToReal(ptArray)[:XEnc.InnerRows*XEnc.InnerCols]
			res := plainUtils.TransposeDense(mat.NewDense(XEnc.InnerCols, XEnc.InnerRows, resReal))
			// now this is x
			Xb.Blocks[i][j] = res
		}
	}
	return plainUtils.MatToArray(plainUtils.ExpandBlocks(Xb))
}

//Decrypts input without decoding nor further transformation, i.e applies transformation:
//Block_ct(i,j) --> Block_pt(i,j)
func DecInputNoDecode(XEnc *EncInput, Box CkksBox) *PlainInput {

	Xb := new(PlainInput)
	Xb.RowP = XEnc.RowP
	Xb.ColP = XEnc.ColP
	Xb.InnerRows = XEnc.InnerRows
	Xb.InnerCols = XEnc.InnerCols
	Xb.Blocks = make([][]*ckks.Plaintext, Xb.RowP)
	for i := 0; i < XEnc.RowP; i++ {
		Xb.Blocks[i] = make([]*ckks.Plaintext, Xb.ColP)
		for j := 0; j < XEnc.ColP; j++ {
			Xb.Blocks[i][j] = Box.Decryptor.DecryptNew(XEnc.Blocks[i][j])
		}
	}
	return Xb
}

//	Given a block input matrix, decrypts and returns the underlying original matrix
//	The sub-matrices are also transposed (remember that they are in form flatten(A.T))
func DecodeInput(XEnc *PlainInput, Box CkksBox) [][]float64 {
	Xb := new(plainUtils.BMatrix)
	Xb.RowP = XEnc.RowP
	Xb.ColP = XEnc.ColP
	Xb.InnerRows = XEnc.InnerRows
	Xb.InnerCols = XEnc.InnerCols
	Xb.Blocks = make([][]*mat.Dense, Xb.RowP)
	for i := 0; i < XEnc.RowP; i++ {
		Xb.Blocks[i] = make([]*mat.Dense, Xb.ColP)
		for j := 0; j < XEnc.ColP; j++ {
			ptArray := Box.Encoder.DecodeSlots(XEnc.Blocks[i][j], Box.Params.LogSlots())
			//this is flatten(x.T)
			resReal := plainUtils.ComplexToReal(ptArray)[:XEnc.InnerRows*XEnc.InnerCols]
			res := plainUtils.TransposeDense(mat.NewDense(XEnc.InnerCols, XEnc.InnerRows, resReal))
			// now this is x
			Xb.Blocks[i][j] = res
		}
	}
	return plainUtils.MatToArray(plainUtils.ExpandBlocks(Xb))
}

//Return encrypted weight in block matrix form. The matrix is also block-transposed
func NewEncWeightDiag(Wm *mat.Dense, rowP, colP, leftInnerDim int, level int, Box CkksBox) (*EncWeightDiag, error) {
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	Wbt := plainUtils.TransposeBlocks(Wb)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	WEnc := new(EncWeightDiag)
	WEnc.RowP = Wbt.RowP
	WEnc.ColP = Wbt.ColP
	WEnc.LeftDim = leftInnerDim
	WEnc.InnerRows = Wb.InnerRows
	WEnc.InnerCols = Wb.InnerCols
	WEnc.Blocks = make([][]*EncDiagMat, Wbt.RowP)
	for i := 0; i < Wbt.RowP; i++ {
		WEnc.Blocks[i] = make([]*EncDiagMat, Wbt.ColP)
		for j := 0; j < Wbt.ColP; j++ {
			//leftDim has to be the rows of EncInput sub matrices
			WEnc.Blocks[i][j] = new(EncDiagMat)
			WEnc.Blocks[i][j].Diags = EncryptWeights(level, plainUtils.MatToArray(Wbt.Blocks[i][j]), leftInnerDim, Box)
			WEnc.NumDiags = len(WEnc.Blocks[i][j].Diags)
		}
	}
	utils.ThrowErr(err)
	return WEnc, nil
}

//Return plaintex weight in block matrix form. The matrix is also block-transposed
func NewPlainWeightDiag(Wm *mat.Dense, rowP, colP, leftInnerDim int, level int, Box CkksBox) (*PlainWeightDiag, error) {
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	Wbt := plainUtils.TransposeBlocks(Wb)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	Wp := new(PlainWeightDiag)
	Wp.RowP = Wbt.RowP
	Wp.ColP = Wbt.ColP
	Wp.LeftDim = leftInnerDim
	Wp.InnerRows = Wbt.InnerRows
	Wp.InnerCols = Wbt.InnerCols

	Wp.Blocks = make([][]*PlainDiagMat, Wbt.RowP)
	for i := 0; i < Wbt.RowP; i++ {
		Wp.Blocks[i] = make([]*PlainDiagMat, Wbt.ColP)
		for j := 0; j < Wbt.ColP; j++ {
			//leftDim has to be the rows of EncInput sub matrices
			Wp.Blocks[i][j] = new(PlainDiagMat)
			Wp.Blocks[i][j].Diags = EncodeWeights(level, plainUtils.MatToArray(Wbt.Blocks[i][j]), leftInnerDim, Box)
			Wp.NumDiags = len(Wp.Blocks[i][j].Diags)
		}
	}
	utils.ThrowErr(err)
	return Wp, nil
}

func (W *EncWeightDiag) GetBlock(i, j int) []ckks.Operand {
	v := make([]ckks.Operand, len(W.Blocks[i][j].Diags))
	for k := range v {
		v[k] = W.Blocks[i][j].Diags[k]
	}
	return v
}

func (W *EncWeightDiag) GetPartitions() (int, int) {
	return W.RowP, W.ColP
}

func (W *EncWeightDiag) GetInnerDims() (int, int) {
	return W.InnerRows, W.InnerCols
}

func (W *EncWeightDiag) Level() int {
	return W.GetBlock(0, 0)[0].Level()
}

func (W *EncWeightDiag) Scale() float64 {
	return W.GetBlock(0, 0)[0].ScalingFactor()
}

func (W *PlainWeightDiag) GetBlock(i, j int) []ckks.Operand {
	v := make([]ckks.Operand, len(W.Blocks[i][j].Diags))
	for k := range v {
		v[k] = W.Blocks[i][j].Diags[k]
	}
	return v
}

func (W *PlainWeightDiag) GetPartitions() (int, int) {
	return W.RowP, W.ColP
}

func (W *PlainWeightDiag) GetInnerDims() (int, int) {
	return W.InnerRows, W.InnerCols
}

func (W *PlainWeightDiag) Level() int {
	return W.GetBlock(0, 0)[0].Level()
}

func (W *PlainWeightDiag) Scale() float64 {
	return W.GetBlock(0, 0)[0].ScalingFactor()
}

//masks input
func MaskInput(Xenc *EncInput, Box CkksBox) *PlainInput {
	mask := make([][]*ckks.Plaintext, len(Xenc.Blocks))
	for i := range Xenc.Blocks {
		mask[i] = make([]*ckks.Plaintext, len(Xenc.Blocks[i]))
		for j := range Xenc.Blocks[i] {
			m := plainUtils.SecureRandMask(Xenc.InnerRows*Xenc.InnerCols, Box.Params.DefaultScale(), Box.Params.QiFloat64(0))
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

//mask input with lambda bit security
func MaskInputV2(Xenc *EncInput, Box CkksBox, lambda int) *PlainInput {
	ringQ := Box.Params.RingQ()
	level := Xenc.Blocks[0][0].Level()
	scale := Xenc.Blocks[0][0].Scale
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

	dslots := 1 << Box.Params.LogSlots()
	//if ringQ.Type() == ring.Standard {
	//	dslots *= 2
	//}

	// Generate the mask in Z[Y] for Y = X^{N/(2*slots)}
	maskBlocks := make([][]*ckks.Plaintext, Xenc.RowP)
	for i := 0; i < Xenc.RowP; i++ {
		maskBlocks[i] = make([]*ckks.Plaintext, Xenc.ColP)
		for j := 0; j < Xenc.ColP; j++ {
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
			ckks.NttAndMontgomeryLvl(levelQ, Box.Params.LogSlots(), ringQ, false, maskPoly)
			maskPt := ckks.NewPlaintext(Box.Params, level, scale)
			maskPt.Value = maskPoly
			maskPt.Value.IsNTT = true
			maskBlocks[i][j] = maskPt

			maskPoly = ringQ.NewPoly()
			ringQ.SetCoefficientsBigintLvl(levelQ, maskBigintNeg[:dslots], maskPoly)
			ckks.NttAndMontgomeryLvl(levelQ, Box.Params.LogSlots(), ringQ, false, maskPoly)
			maskPtNeg := ckks.NewPlaintext(Box.Params, level, scale)
			maskPtNeg.Value = maskPoly
			maskPtNeg.Value.IsNTT = true

			Box.Evaluator.Add(Xenc.Blocks[i][j], maskPtNeg, Xenc.Blocks[i][j])
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

//removes mask from MaskInputV2
func UnmaskInput(Xmask, mask *PlainInput, Box CkksBox) {
	//ephemeral secret key
	kgen := ckks.NewKeyGenerator(Box.Params)
	skEph := kgen.GenSecretKey()
	for i := 0; i < Xmask.RowP; i++ {
		for j := 0; j < Xmask.ColP; j++ {
			ct := Box.Encryptor.WithKey(skEph).EncryptNew(Xmask.Blocks[i][j])
			Box.Evaluator.Add(ct, mask.Blocks[i][j], ct)
			Box.Decryptor.WithKey(skEph).Decrypt(ct, Xmask.Blocks[i][j])
		}
	}
}
