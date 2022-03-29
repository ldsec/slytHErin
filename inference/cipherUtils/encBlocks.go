package cipherUtils

import "github.com/tuneinsight/lattigo/v3/ckks"

type EncInput struct {
	//stores an encrypted input block matrix
	Blocks     [][]*ckks.Ciphertext //all the sub-matrixes, encrypted as flatten(A.T)
	RowP, ColP int                  //num of partitions
	InnerLen   int                  //size of flatten sub-matrixes
}
type EncDiagMat struct {
	//store an encrypted weight matrix in diagonal form
	Diags    []*ckks.Ciphertext //enc diagonals
	LeftDim  int                //rows of left matrix
	NumDiags int
}
type EncWeight struct {
	Blocks     [][]*EncDiagMat //blocks of the matrix, each is a sub-matrix in diag form
	RowP, ColP int
}
