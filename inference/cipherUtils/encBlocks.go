package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
)

type EncInput struct {
	//stores an encrypted input block matrix
	Blocks     [][]*ckks.Ciphertext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP int                  //num of partitions
	InnerRows  int                  //rows of sub-matrix
	InnerCols  int
}
type PlainInput struct {
	//stores a plaintext input block matrix --> this is used for addition ops
	Blocks     [][]*ckks.Plaintext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP int                 //num of partitions
	InnerRows  int                 //rows of sub-matrix
	InnerCols  int
}

type EncDiagMat struct {
	//store an encrypted weight matrix in diagonal form
	Diags []*ckks.Ciphertext //enc diagonals
}
type EncWeightDiag struct {
	Blocks     [][]*EncDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP int
	LeftDim    int //rows of left matrix
	NumDiags   int
	InnerRows  int //rows of matrix
	InnerCols  int
}
type PlainDiagMat struct {
	//store a plaintext weight matrix in diagonal form
	Diags []*ckks.Plaintext //diagonals
}
type PlainWeightDiag struct {
	Blocks     [][]*PlainDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP int
	LeftDim    int //rows of left matrix
	NumDiags   int
	InnerRows  int //rows of matrix
	InnerCols  int
}

func NewPlainInput(X [][]float64, rowP, colP int, level int, Box CkksBox) (*PlainInput, error) {
	Xm := plainUtils.NewDense(X)
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
			XPlain.Blocks[i][j] = EncodeInput(level, plainUtils.MatToArray(Xb.Blocks[i][j]), Box)
		}
	}
	return XPlain, nil
}

func NewEncInput(X [][]float64, rowP, colP int, level int, Box CkksBox) (*EncInput, error) {
	Xm := plainUtils.NewDense(X)
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
		utils.ThrowErr(errors.New("Input submatrixes elements must be less than 2^(LogSlots-1)"))
	}
	XEnc.Blocks = make([][]*ckks.Ciphertext, rowP)
	for i := 0; i < rowP; i++ {
		XEnc.Blocks[i] = make([]*ckks.Ciphertext, colP)
		for j := 0; j < colP; j++ {
			XEnc.Blocks[i][j] = EncryptInput(level, plainUtils.MatToArray(Xb.Blocks[i][j]), Box)
		}
	}
	return XEnc, nil
}

func NewEncWeightDiag(W [][]float64, rowP, colP, leftInnerDim int, level int, Box CkksBox) (*EncWeightDiag, error) {
	Wm := plainUtils.NewDense(W)
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	WEnc := new(EncWeightDiag)
	WEnc.RowP = rowP
	WEnc.ColP = colP
	WEnc.LeftDim = leftInnerDim
	WEnc.InnerRows = Wb.InnerRows
	WEnc.InnerCols = Wb.InnerCols
	WEnc.Blocks = make([][]*EncDiagMat, rowP)
	for i := 0; i < rowP; i++ {
		WEnc.Blocks[i] = make([]*EncDiagMat, colP)
		for j := 0; j < colP; j++ {
			//leftDim has to be the rows of EncInput sub matrices
			WEnc.Blocks[i][j] = new(EncDiagMat)
			WEnc.Blocks[i][j].Diags = EncryptWeights(level, plainUtils.MatToArray(Wb.Blocks[i][j]), leftInnerDim, Box)
			WEnc.NumDiags = len(WEnc.Blocks[i][j].Diags)
		}
	}
	utils.ThrowErr(err)
	return WEnc, nil
}

func NewPlainWeightDiag(W [][]float64, rowP, colP, leftInnerDim int, level int, Box CkksBox) (*PlainWeightDiag, error) {
	Wm := plainUtils.NewDense(W)
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	if err != nil {
		utils.ThrowErr(err)
		return nil, err
	}
	Wp := new(PlainWeightDiag)
	Wp.RowP = rowP
	Wp.ColP = colP
	Wp.LeftDim = leftInnerDim
	Wp.InnerRows = Wb.InnerRows
	Wp.InnerCols = Wb.InnerCols
	Wp.Blocks = make([][]*PlainDiagMat, rowP)
	for i := 0; i < rowP; i++ {
		Wp.Blocks[i] = make([]*PlainDiagMat, colP)
		for j := 0; j < colP; j++ {
			//leftDim has to be the rows of EncInput sub matrices
			Wp.Blocks[i][j] = new(PlainDiagMat)
			Wp.Blocks[i][j].Diags = EncodeWeights(level, plainUtils.MatToArray(Wb.Blocks[i][j]), leftInnerDim, Box)
			Wp.NumDiags = len(Wp.Blocks[i][j].Diags)
		}
	}
	utils.ThrowErr(err)
	return Wp, nil
}

func GetOptimalInnerRows(inputInnerCols int, params ckks.Parameters) int {
	slotsAvailable := float64(math.Pow(2, float64(params.LogSlots()-1)))
	return int(math.Floor(slotsAvailable / float64(inputInnerCols)))
}
