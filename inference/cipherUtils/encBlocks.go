package cipherUtils

import (
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
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

func NewPlainInput(X [][]float64, rowP, colP int, Box CkksBox) (*PlainInput, error) {
	Xm := plainUtils.NewDense(X)
	Xb, err := plainUtils.PartitionMatrix(Xm, rowP, colP)
	if err != nil {
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
			XPlain.Blocks[i][j] = EncodeInput(Box.Params.MaxLevel(), plainUtils.MatToArray(Xb.Blocks[i][j]), Box)
		}
	}
	return XPlain, nil
}

func NewEncInput(X [][]float64, rowP, colP int, Box CkksBox) (*EncInput, error) {
	Xm := plainUtils.NewDense(X)
	Xb, err := plainUtils.PartitionMatrix(Xm, rowP, colP)
	if err != nil {
		return nil, err
	}
	XEnc := new(EncInput)
	XEnc.RowP = rowP
	XEnc.ColP = colP
	XEnc.InnerRows = Xb.InnerRows
	XEnc.InnerCols = Xb.InnerCols
	XEnc.Blocks = make([][]*ckks.Ciphertext, rowP)
	for i := 0; i < rowP; i++ {
		XEnc.Blocks[i] = make([]*ckks.Ciphertext, colP)
		for j := 0; j < colP; j++ {
			XEnc.Blocks[i][j] = EncryptInput(Box.Params.MaxLevel(), plainUtils.MatToArray(Xb.Blocks[i][j]), Box)
		}
	}
	return XEnc, nil
}

func DecInput(XEnc *EncInput, Box CkksBox) [][]float64 {
	/*
		Given a block input matrix, decrypts and returns the underlying original matrix
		The sub-matrices are also transposed (remember that they are in form flatten(A.T))
	*/
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
			Xb.Blocks[i][j] = res
		}
	}
	return plainUtils.MatToArray(plainUtils.ExpandBlocks(Xb))
}

func NewEncWeightDiag(W [][]float64, rowP, colP, leftInnerDim int, Box CkksBox) (*EncWeightDiag, error) {
	Wm := plainUtils.NewDense(W)
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	if err != nil {
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
			WEnc.Blocks[i][j].Diags = EncryptWeights(Box.Params.MaxLevel(), plainUtils.MatToArray(Wb.Blocks[i][j]), leftInnerDim, Box)
			WEnc.NumDiags = len(WEnc.Blocks[i][j].Diags)
		}
	}
	return WEnc, nil
}

func NewPlainWeightDiag(W [][]float64, rowP, colP, leftInnerDim int, Box CkksBox) (*PlainWeightDiag, error) {
	Wm := plainUtils.NewDense(W)
	Wb, err := plainUtils.PartitionMatrix(Wm, rowP, colP)
	if err != nil {
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
			Wp.Blocks[i][j].Diags = EncodeWeights(Box.Params.MaxLevel(), plainUtils.MatToArray(Wb.Blocks[i][j]), leftInnerDim, Box)
			Wp.NumDiags = len(Wp.Blocks[i][j].Diags)
		}
	}
	return Wp, nil
}
