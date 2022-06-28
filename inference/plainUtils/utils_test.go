package plainUtils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestBlock(t *testing.T) {
	m := RandMatrix(5, 4)
	bm, err := PartitionMatrix(m, 2, 2)
	if err != nil {
		panic(err)
	}
	m2 := ExpandBlocks(bm)
	for i := 0; i < NumRows(m); i++ {
		fmt.Println("real:", m.RawRowView(i))
		fmt.Println("test:", m2.RawRowView(i))
	}
	fmt.Println("Distance:", Distance(RowFlatten(m), RowFlatten(m2)))
}

func TestSumBlock(t *testing.T) {
	m := RandMatrix(64, 845)
	for i := 0; i < NumRows(m); i++ {
		fmt.Println(m.RawRowView(i))
	}
	bm, err := PartitionMatrix(m, 1, 65)
	if err != nil {
		panic(err)
	}
	bmS, err := AddBlocks(bm, bm)
	var mS mat.Dense
	mS.Add(m, m)

	fmt.Println("expected:")
	for i := 0; i < NumRows(&mS); i++ {
		fmt.Println(mS.RawRowView(i))
	}
	fmt.Println("blocks:")
	m = ExpandBlocks(bmS)
	for i := 0; i < NumRows(m); i++ {
		fmt.Println(m.RawRowView(i))
	}
	fmt.Println("Distance:", Distance(RowFlatten(m), RowFlatten(&mS)))
}

func TestMultiPlyBlocks(t *testing.T) {
	a := RandMatrix(63, 64)
	b := RandMatrix(64, 64)

	ba, err := PartitionMatrix(a, 2, 2)
	bb, err := PartitionMatrix(b, 2, 2)
	var c mat.Dense
	c.Mul(a, b)
	bc, err := MultiPlyBlocks(ba, bb)
	if err != nil {
		panic(err)
	}
	fmt.Println("expected:")
	for i := 0; i < NumRows(&c); i++ {
		fmt.Println(c.RawRowView(i))
	}
	fmt.Println("blocks:")
	m := ExpandBlocks(bc)
	for i := 0; i < NumRows(m); i++ {
		fmt.Println(m.RawRowView(i))
	}
	fmt.Println("Distance:", Distance(RowFlatten(m)[:NumRows(&c)*NumCols(&c)], RowFlatten(&c)))
}
