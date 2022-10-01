package cipherUtils

import (
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Interface for generic block matrix. In can be a plaintext or ciphertext matrix
type BlocksOperand interface {
	GetBlock(i, j int) interface{}
	GetPartitions() (int, int)
	GetInnerDims() (int, int)
	GetRealDims() (int, int)
	Level() int
	Scale() float64
}

//Encrypted block matrix, to be used for input or bias layer
type EncInput struct {
	Blocks             [][]*ckks.Ciphertext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP         int                  //num of partitions
	InnerRows          int                  //rows of sub-matrix
	InnerCols          int
	RealRows, RealCols int
}

//Plaintext block matrix, to be used for input or bias layer
type PlainInput struct {
	Blocks             [][]*ckks.Plaintext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP         int                 //num of partitions
	InnerRows          int                 //rows of sub-matrix
	InnerCols          int
	RealRows, RealCols int
}

type DiagMat interface {
	GetDiags() []ckks.Operand
}

//Encrypted matrix in diagonal form
type EncDiagMat struct {
	//store an encrypted weight matrix in diagonal form
	Diags        []*ckks.Ciphertext //enc diagonals
	InnerRows    int                //rows of sub-matrix
	InnerCols    int
	LeftR, LeftC int //rows cols of left matrix
}

func (W *EncDiagMat) GetRotations(params ckks.Parameters) []int {
	var rotations = []int{W.LeftR}
	for i := 1; i < (W.InnerRows+1)>>1; i++ {
		r := 2 * i * W.LeftR
		rotations = append(rotations, r)
	}
	rotations = append(rotations, W.InnerRows)
	rotations = append(rotations, -W.LeftR*W.InnerRows)
	rotations = append(rotations, -2*W.LeftR*W.InnerRows)
	if W.InnerRows < W.InnerCols {
		replicationFactor := GetReplicaFactor(W.InnerRows, W.InnerCols)
		rotations = append(rotations, params.RotationsForReplicateLog(W.LeftR*W.LeftC, replicationFactor)...)
	}
	return rotations
}

func (W *EncDiagMat) GetDiags() []ckks.Operand {
	diags := make([]ckks.Operand, len(W.Diags))
	for i := range diags {
		diags[i] = W.Diags[i]
	}
	return diags
}

//Encrypted block matrix, weight of dense or convolutional layer
type EncWeightDiag struct {
	Blocks             [][]*EncDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP         int
	LeftR, LeftC       int //rows cols of left matrix
	InnerRows          int //rows of matrix
	InnerCols          int
	RealRows, RealCols int
}

//Plaintext matrix in diagonal form
type PlainDiagMat struct {
	//store a plaintext weight matrix in diagonal form
	Diags                []*ckks.Plaintext //diagonals
	InnerRows, InnerCols int
	LeftR, LeftC         int //rows cols of left matrix
}

func (W *PlainDiagMat) GetRotations(params ckks.Parameters) []int {
	var rotations = []int{W.LeftR}
	for i := 1; i < (W.InnerRows+1)>>1; i++ {
		r := 2 * i * W.LeftR
		rotations = append(rotations, r)
	}
	rotations = append(rotations, W.InnerRows)
	rotations = append(rotations, -W.LeftR*W.InnerRows)
	rotations = append(rotations, -2*W.LeftR*W.InnerRows)
	if W.InnerRows < W.InnerCols {
		replicationFactor := GetReplicaFactor(W.InnerRows, W.InnerCols)
		rotations = append(rotations, params.RotationsForReplicateLog(W.LeftR*W.LeftC, replicationFactor)...)
	}
	return rotations
}

func (W *PlainDiagMat) GetDiags() []ckks.Operand {
	diags := make([]ckks.Operand, len(W.Diags))
	for i := range diags {
		diags[i] = W.Diags[i]
	}
	return diags
}

//Plaintext block matrix, weight of dense or convolutional layer
type PlainWeightDiag struct {
	Blocks             [][]*PlainDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP         int
	LeftR, LeftC       int //rows of left matrix
	InnerRows          int //rows of matrix
	InnerCols          int
	RealRows, RealCols int
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
	XPlain.RealRows, XPlain.RealCols = Xm.Dims()
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
	XEnc.RealRows, XEnc.RealCols = Xm.Dims()
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

func (X *EncInput) GetBlock(i, j int) interface{} {
	return X.Blocks[i][j]
}

func (X *EncInput) GetPartitions() (int, int) {
	return X.RowP, X.ColP
}

func (X *EncInput) GetInnerDims() (int, int) {
	return X.InnerRows, X.InnerCols
}

func (X *EncInput) GetRealDims() (int, int) {
	return X.RealRows, X.RealCols
}

func (X *EncInput) Level() int {
	return X.GetBlock(0, 0).([]*ckks.Ciphertext)[0].Level()
}

func (X *EncInput) Scale() float64 {
	return X.GetBlock(0, 0).([]*ckks.Ciphertext)[0].ScalingFactor()
}

func (X *PlainInput) GetBlock(i, j int) interface{} {
	return X.Blocks[i][j]
}

func (X *PlainInput) GetRealDims() (int, int) {
	return X.RealRows, X.RealCols
}

func (X *PlainInput) GetPartitions() (int, int) {
	return X.RowP, X.ColP
}

func (X *PlainInput) GetInnerDims() (int, int) {
	return X.InnerRows, X.InnerCols
}

func (X *PlainInput) Level() int {
	return X.GetBlock(0, 0).([]*ckks.Plaintext)[0].Level()
}

func (X *PlainInput) Scale() float64 {
	return X.GetBlock(0, 0).([]*ckks.Plaintext)[0].ScalingFactor()
}

//	Given a block input matrix, decrypts and returns the underlying original matrix
//	The sub-matrices are also transposed (remember that they are in form flatten(A.T))
func DecInput(XEnc *EncInput, Box CkksBox) [][]float64 {
	Xb := new(plainUtils.BMatrix)
	Xb.RealRows = XEnc.RealRows
	Xb.RealCols = XEnc.RealCols
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
	Xb.RealRows = XEnc.RealRows
	Xb.RealCols = XEnc.RealCols
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
	Xb.RealRows = XEnc.RealRows
	Xb.RealCols = XEnc.RealCols
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
//i.e the first column of block is stored as row for cache efficiency
//must provide the partitions and the rows and cols of the input inner sumbatrices (as when it will be multiplied)
func NewEncWeightDiag(Wm *mat.Dense, rowP, colP, leftR, leftC int, level int, Box CkksBox) (*EncWeightDiag, error) {
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	Wbt := plainUtils.TransposeBlocks(Wb)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	WEnc := new(EncWeightDiag)
	WEnc.RowP = Wbt.RowP
	WEnc.ColP = Wbt.ColP
	WEnc.LeftR, WEnc.LeftC = leftR, leftC
	WEnc.InnerRows = Wb.InnerRows
	WEnc.InnerCols = Wb.InnerCols
	WEnc.RealRows, WEnc.RealCols = Wm.Dims()
	WEnc.Blocks = make([][]*EncDiagMat, Wbt.RowP)
	for i := 0; i < Wbt.RowP; i++ {
		WEnc.Blocks[i] = make([]*EncDiagMat, Wbt.ColP)
		for j := 0; j < Wbt.ColP; j++ {
			//LeftR has to be the rows of EncInput sub matrices
			WEnc.Blocks[i][j] = new(EncDiagMat)
			WEnc.Blocks[i][j] = EncryptWeights(level, plainUtils.MatToArray(Wbt.Blocks[i][j]), leftR, leftC, Box)
		}
	}
	utils.ThrowErr(err)
	return WEnc, nil
}

//Return plaintex weight in block matrix form. The matrix is also block-transposed
//i.e the first column of block is stored as row for cache efficiency
//takes block partions, rows of input inner submatrices, level, whether to use the complex trick, and box
func NewPlainWeightDiag(Wm *mat.Dense, rowP, colP, leftR, leftC int, level int, Box CkksBox) (*PlainWeightDiag, error) {
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	Wbt := plainUtils.TransposeBlocks(Wb)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	Wp := new(PlainWeightDiag)
	Wp.RealRows, Wp.RealCols = Wm.Dims()
	Wp.RowP = Wbt.RowP
	Wp.ColP = Wbt.ColP
	Wp.LeftR, Wp.LeftC = leftR, leftC
	Wp.InnerRows = Wbt.InnerRows
	Wp.InnerCols = Wbt.InnerCols

	Wp.Blocks = make([][]*PlainDiagMat, Wbt.RowP)
	for i := 0; i < Wbt.RowP; i++ {
		Wp.Blocks[i] = make([]*PlainDiagMat, Wbt.ColP)
		for j := 0; j < Wbt.ColP; j++ {
			//LeftR has to be the rows of EncInput sub matrices
			Wp.Blocks[i][j] = new(PlainDiagMat)
			Wp.Blocks[i][j] = EncodeWeights(level, plainUtils.MatToArray(Wbt.Blocks[i][j]), leftR, leftC, Box)
		}
	}
	utils.ThrowErr(err)
	return Wp, nil
}

func (W *EncWeightDiag) GetBlock(i, j int) interface{} {
	return W.Blocks[i][j]
}

func (W *EncWeightDiag) GetPartitions() (int, int) {
	return W.RowP, W.ColP
}

func (W *EncWeightDiag) GetInnerDims() (int, int) {
	return W.InnerRows, W.InnerCols
}

func (W *EncWeightDiag) GetRealDims() (int, int) {
	return W.RealRows, W.RealCols
}

func (W *EncWeightDiag) Level() int {
	return W.GetBlock(0, 0).([]*ckks.Ciphertext)[0].Level()
}

func (W *EncWeightDiag) Scale() float64 {
	return W.GetBlock(0, 0).([]*ckks.Ciphertext)[0].ScalingFactor()
}

func (W *EncWeightDiag) GetRotations(params ckks.Parameters) []int {
	return W.Blocks[0][0].GetRotations(params)
}

func (W *PlainWeightDiag) GetBlock(i, j int) interface{} {
	return W.Blocks[i][j]
}

func (W *PlainWeightDiag) GetPartitions() (int, int) {
	return W.RowP, W.ColP
}

func (W *PlainWeightDiag) GetInnerDims() (int, int) {
	return W.InnerRows, W.InnerCols
}

func (W *PlainWeightDiag) GetRealDims() (int, int) {
	return W.RealRows, W.RealCols
}

func (W *PlainWeightDiag) Level() int {
	return W.GetBlock(0, 0).(*ckks.LinearTransform).Level
}

func (W *PlainWeightDiag) Scale() float64 {
	return W.GetBlock(0, 0).(*ckks.LinearTransform).Scale
}

func (W *PlainWeightDiag) GetRotations(params ckks.Parameters) []int {
	return W.Blocks[0][0].GetRotations(params)
}
