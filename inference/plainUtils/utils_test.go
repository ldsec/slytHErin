package plainUtils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestRotateRealArray(t *testing.T) {
	v := make([]float64, 9)

	for i := range v {
		v[i] = float64(i) + 1.0
	}
	fmt.Println(v)
	fmt.Println(RotateRealArray(v, -4))
	for i := range v {
		v[i] = float64(i) + 1.0
	}
	fmt.Println(RotateRealArray(v, 4))
}

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

func TestBlockWPad(t *testing.T) {
	m := RandMatrix(128, 128)
	PrintDense(m)
	bm, err := PartitionMatrix(m, 2, 2)
	if err != nil {
		panic(err)
	}
	PrintBlocks(bm)
	m2 := ExpandBlocks(bm)
	for i := 0; i < NumRows(m); i++ {
		fmt.Println("real:", m.RawRowView(i))
		fmt.Println("test:", m2.RawRowView(i))
	}
	fmt.Println("Distance:", Distance(RowFlatten(m), RowFlatten(m2)))

	w1 := RandMatrix(128, 128)
	w2 := RandMatrix(128, 186)
	w3 := RandMatrix(186, 150)

	wm1, err := PartitionMatrix(w1, 2, 2)
	wm2, err := PartitionMatrixSquare(w2, 2, 3, 128/2) //mul, thus square
	wm3, err := PartitionMatrixSquare(w3, 3, 3, 128/2) //mul, thus square
	//wm2, err := PartitionMatrix(w2, 2, 2)
	PrintBlocks(wm2)
	a, _ := AddBlocks(bm, wm1)
	tmp1 := new(mat.Dense)
	tmp1.Add(m, w1)
	r := ExpandBlocks(a)
	fmt.Println("Distance:", Distance(RowFlatten(r), RowFlatten(tmp1)))
	b, _ := MultiPlyBlocks(a, wm2)
	PrintBlocks(b)
	r = ExpandBlocks(b)
	PrintDense(r)
	fmt.Println("_____________________________________________")
	tmp2 := new(mat.Dense)
	tmp2.Mul(tmp1, w2)
	PrintDense(tmp2)
	fmt.Println("Distance:", Distance(RowFlatten(r), RowFlatten(tmp2)))
	c, _ := MultiPlyBlocks(b, wm3)
	PrintBlocks(c)
	r = ExpandBlocks(c)
	PrintDense(r)
	fmt.Println("_____________________________________________")
	tmp3 := new(mat.Dense)
	tmp3.Mul(tmp2, w3)
	PrintDense(tmp3)
	fmt.Println("Distance:", Distance(RowFlatten(r), RowFlatten(tmp3))/(128*150))
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

func TestMultiPlyBlocks2(t *testing.T) {
	m := RandMatrix(128, 128)
	PrintDense(m)
	bm, err := PartitionMatrix(m, 2, 2)
	if err != nil {
		panic(err)
	}
	PrintBlocks(bm)
	m2 := ExpandBlocks(bm)
	for i := 0; i < NumRows(m); i++ {
		fmt.Println("real:", m.RawRowView(i))
		fmt.Println("test:", m2.RawRowView(i))
	}
	fmt.Println("Distance:", Distance(RowFlatten(m), RowFlatten(m2)))

	w1 := RandMatrix(128, 128)
	w2 := RandMatrix(128, 128)

	wm1, err := PartitionMatrix(w1, 2, 2)
	//wm2, err := PartitionMatrixSquare(w2, 2, 2, 128/2) //mul, thus square
	wm2, err := PartitionMatrix(w2, 2, 2)
	PrintBlocks(wm2)
	a, _ := AddBlocks(bm, wm1)
	tmp1 := new(mat.Dense)
	tmp1.Add(m, w1)
	r := ExpandBlocks(a)
	fmt.Println("Distance:", Distance(RowFlatten(r), RowFlatten(tmp1)))
	b, _ := MultiPlyBlocks(a, wm2)
	PrintBlocks(b)
	r = ExpandBlocks(b)
	PrintDense(r)
	fmt.Println("_____________________________________________")
	tmp2 := new(mat.Dense)
	tmp2.Mul(tmp1, w2)
	PrintDense(tmp2)
	fmt.Println("Distance:", Distance(RowFlatten(r), RowFlatten(tmp2)))
}
