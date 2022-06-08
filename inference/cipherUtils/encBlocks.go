package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"gonum.org/v1/gonum/mat"
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
			// now this is x
			Xb.Blocks[i][j] = res
		}
	}
	return plainUtils.MatToArray(plainUtils.ExpandBlocks(Xb))
}

func NewEncWeightDiag(W [][]float64, rowP, colP, leftInnerDim int, level int, Box CkksBox) (*EncWeightDiag, error) {
	Wm := plainUtils.NewDense(W)
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

func NewPlainWeightDiag(W [][]float64, rowP, colP, leftInnerDim int, level int, Box CkksBox) (*PlainWeightDiag, error) {
	Wm := plainUtils.NewDense(W)
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
	//add safety check for dimentions (dimMid > dimOut -> 2x space o.w 3x space)
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

//Computes the optimal number of rows for input sub-matrices. Takes the innerCols of the Input and the maxInnerCols of all the weights in the pipeline
func GetOptimalInnerRows(inputInnerCols int, maxInnerCols int, params ckks.Parameters) int {
	innerCols := plainUtils.Max(inputInnerCols, maxInnerCols)
	slotsAvailable := float64(math.Pow(2, float64(params.LogSlots()-1)))
	optInnerRows := int(math.Floor(slotsAvailable / float64(innerCols)))
	//takes into account that if maxInnerCols > inputInnerRows we will have to rotate during prepacking with 3x space occupied
	if optInnerRows*inputInnerCols*3 > params.LogSlots() {
		return optInnerRows
	} else {
		for optInnerRows*inputInnerCols*3 > params.LogSlots() {
			optInnerRows--
		}
		return optInnerRows
	}
}
