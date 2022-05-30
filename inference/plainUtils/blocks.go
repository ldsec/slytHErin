package plainUtils

import (
	"errors"
	"gonum.org/v1/gonum/mat"
	"sync"
)

type BMatrix struct {
	//block matrix
	Blocks               [][]*mat.Dense
	RowP, ColP           int //num of partitions
	InnerRows, InnerCols int //size of sub-matrixes
}

func TransposeBlocks(Bm *BMatrix) *BMatrix {
	Tm := new(BMatrix)
	Tm.RowP = Bm.ColP
	Tm.ColP = Bm.RowP
	Tm.InnerRows, Tm.InnerCols = Bm.InnerRows, Bm.InnerCols
	Tm.Blocks = make([][]*mat.Dense, Tm.RowP)
	for i := 0; i < Tm.RowP; i++ {
		Tm.Blocks[i] = make([]*mat.Dense, Tm.ColP)
		for j := 0; j < Tm.ColP; j++ {
			Tm.Blocks[i][j] = Bm.Blocks[j][i]
		}
	}
	return Tm
}
func PartitionMatrix(m *mat.Dense, rowP, colP int) (*BMatrix, error) {
	/*
		Partitions m into a rowPxcolP Block Matrix
		where each sub-matrix is row(m)/rowP x col(m)/colP
	*/
	rowM, colM := m.Dims()
	if rowM%rowP != 0 || colM%colP != 0 {
		return nil, errors.New("Cannot Split Matrix in Blocks!")
	}
	rowS := rowM / rowP
	colS := colM / colP
	Bm := make([][]*mat.Dense, rowP)
	for i := 0; i < rowP; i++ {
		Bm[i] = make([]*mat.Dense, colP)
		for j := 0; j < colP; j++ {
			Bm[i][j] = mat.NewDense(rowS, colS, nil)
			for s := i * rowS; s < (i+1)*rowS; s++ {
				for r := j * colS; r < (j+1)*colS; r++ {
					Bm[i][j].Set(s%rowS, r%colS, m.At(s, r))
				}
			}
		}
	}
	return &BMatrix{Blocks: Bm, RowP: rowP, ColP: colP, InnerRows: rowS, InnerCols: colS}, nil
}

func ExpandBlocks(Bm *BMatrix) *mat.Dense {
	//Reconstruct a matrix from block representation
	m := mat.NewDense(Bm.RowP*Bm.InnerRows, Bm.ColP*Bm.InnerCols, nil)
	for i := 0; i < NumRows(m); i++ {
		outerI := i / Bm.InnerRows
		innerI := i % Bm.InnerRows
		for j := 0; j < NumCols(m); j++ {
			outerJ := j / Bm.InnerCols
			innerJ := j % Bm.InnerCols
			m.Set(i, j, Bm.Blocks[outerI%Bm.RowP][outerJ%Bm.ColP].At(innerI, innerJ))
		}
	}
	return m
}

func AddBlocks(A, B *BMatrix) (*BMatrix, error) {
	q := A.RowP
	r := B.ColP
	var err error
	if A.RowP != B.RowP || A.ColP != B.ColP {
		err = errors.New("Block partitions not compatible for addition")
	}
	if A.InnerRows != B.InnerRows || A.InnerCols != B.InnerCols {
		err = errors.New("Inner dimensions not compatible for addition")
	}
	C := make([][]*mat.Dense, q)
	for i := 0; i < q; i++ {
		C[i] = make([]*mat.Dense, r)
		for j := 0; j < r; j++ {
			C[i][j] = mat.NewDense(A.InnerRows, A.InnerCols, nil)
			C[i][j].Add(A.Blocks[i][j], B.Blocks[i][j])
		}
	}
	return &BMatrix{Blocks: C, RowP: A.RowP, ColP: B.ColP, InnerRows: A.InnerRows, InnerCols: A.InnerCols}, err
}

func MultiPlyBlocks(A, B *BMatrix) (*BMatrix, error) {
	//reference: https://en.wikipedia.org/wiki/Block_matrix
	q := A.RowP
	r := B.ColP
	var err error
	if A.ColP != B.RowP {
		err = errors.New("Block partitions not compatible for multiplication")
	}
	if A.InnerCols != B.InnerRows {
		err = errors.New("Inner Dimensions not compatible for multiplication")
	}
	s := B.RowP //mid dim
	C := make([][]*mat.Dense, q)

	innerRows := A.InnerRows
	innerCols := B.InnerCols

	var wg sync.WaitGroup

	for i := 0; i < q; i++ {
		C[i] = make([]*mat.Dense, r)
		for j := 0; j < r; j++ {
			partials := make([]*mat.Dense, s)
			for k := 0; k < s; k++ {
				wg.Add(1)
				go func(a, b *mat.Dense, k int, res []*mat.Dense) {
					defer wg.Done()
					cij := mat.NewDense(innerRows, innerCols, nil)
					cij.Mul(a, b)
					res[k] = cij
				}(A.Blocks[i][k], B.Blocks[k][j], k, partials)
			}
			wg.Wait()
			Cij := partials[0]
			for k := 1; k < s; k++ {
				Cij.Add(Cij, partials[k])
			}
			C[i][j] = Cij
		}
	}
	return &BMatrix{Blocks: C, RowP: q, ColP: r, InnerRows: innerRows, InnerCols: innerCols}, err
}

func MultiplyBlocksByConst(A *BMatrix, c float64) *BMatrix {
	newBlocks := make([][]*mat.Dense, A.RowP)
	for i := range newBlocks {
		newBlocks[i] = make([]*mat.Dense, A.ColP)
		for j := range newBlocks[0] {
			newBlocks[i][j] = MulByConst(A.Blocks[i][j], c)
		}
	}
	A.Blocks = newBlocks
	return A
}

func ApplyFunc(Bm *BMatrix, f func(x float64) float64) {
	for i := 0; i < Bm.RowP; i++ {
		for j := 0; j < Bm.ColP; j++ {
			fn := func(i, j int, v float64) float64 {
				return f(v)
			}
			Bm.Blocks[i][j].Apply(fn, Bm.Blocks[i][j])
		}
	}
}
