package multidim

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func MatrixPrint(X mat.Matrix) {
	var empty *mat.Dense
	if X == empty {
		fmt.Printf("[]\n")
		return
	}
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
func ExplicitTranspose(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	ret := mat.NewDense(c, r, nil)
	for i := 0; i < r; i++ {
		ret.SetCol(i, m.(*mat.Dense).RawRowView(i))
	}
	return ret
}
