package multidim

import (
	"fmt"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	utils2 "github.com/ldsec/lattigo/v2/utils"
)

type PackedMatrixMultiplier struct {
	params             ckks2.Parameters
	eval               ckks2.Evaluator
	tmpA, tmpB, tmpC   [2]*ring.Poly
	tmpARescale, tmpAB [][2]*ring.Poly
	tmpBRescale        [][][2]*ring.Poly
	poolDecompQP       []rlwe2.PolyQP
	multiplierMap      map[string]MatrixMultiplication
	transposerMap      map[string]TransposeLT
}

func NewPackedMatrixMultiplier(params ckks2.Parameters, innerDim, maxColumns, maxRows int, eval ckks2.Evaluator) (ppm *PackedMatrixMultiplier) {

	ppm = new(PackedMatrixMultiplier)

	ringQ := params.RingQ()
	ringP := params.RingP()
	level := params.MaxLevel()

	ppm.params = params
	ppm.eval = eval
	ppm.tmpA = [2]*ring.Poly{ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)}
	ppm.tmpB = [2]*ring.Poly{ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)}
	ppm.tmpC = [2]*ring.Poly{ringQ.NewPolyLvl(level - 1), ringQ.NewPolyLvl(level - 1)}
	ppm.tmpAB = make([][2]*ring.Poly, maxColumns)
	for i := range ppm.tmpAB {
		ppm.tmpAB[i] = [2]*ring.Poly{ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)}
	}

	ppm.tmpARescale = make([][2]*ring.Poly, innerDim-1)
	for i := 0; i < innerDim-1; i++ {
		ppm.tmpARescale[i] = [2]*ring.Poly{ringQ.NewPolyLvl(level - 1), ringQ.NewPolyLvl(level - 1)}
	}

	ppm.tmpBRescale = make([][][2]*ring.Poly, maxColumns*maxRows)
	for i := range ppm.tmpBRescale {
		ppm.tmpBRescale[i] = make([][2]*ring.Poly, innerDim-1)
		for j := 0; j < innerDim-1; j++ {
			ppm.tmpBRescale[i][j] = [2]*ring.Poly{ringQ.NewPolyLvl(level - 1), ringQ.NewPolyLvl(level - 1)}
		}
	}

	ppm.poolDecompQP = make([]rlwe2.PolyQP, params.Beta())
	for i := range ppm.poolDecompQP {
		ppm.poolDecompQP[i].Q = ringQ.NewPoly()
		ppm.poolDecompQP[i].P = ringP.NewPoly()
	}

	return
}

func (ppm *PackedMatrixMultiplier) AddMatrixOperation(op interface{}) {
	switch el := op.(type) {
	case MatrixMultiplication:
		if ppm.multiplierMap == nil {
			ppm.multiplierMap = make(map[string]MatrixMultiplication)
		}
		ppm.multiplierMap[el.StringMap()] = el
	case TransposeLT:
		if ppm.transposerMap == nil {
			ppm.transposerMap = make(map[string]TransposeLT)
		}
		ppm.transposerMap[el.StringMap()] = el
	default:
		panic("operation not recognized")
	}
}

func (ppm *PackedMatrixMultiplier) Strassen(A, B *CiphertextBatchMatrix, innerDim int, C *CiphertextBatchMatrix) {

	level := utils2.MinInt(A.Level(), B.Level())

	str := fmt.Sprintf("0x%04o0x%04o", A.dim, level)
	mmpt, ok := ppm.multiplierMap[str]
	if !ok {
		panic(fmt.Sprintf("Multiplier [dimension:%d ; level :%d] missing", A.dim, level))
	}

	params := ppm.params
	eval := ppm.eval
	levelQ := utils2.MinInt(A.Level(), B.Level())
	levelP := params.PCount() - 1
	alpha := params.PCount()

	colsB := B.Cols()

	tmpB := ckks2.NewCiphertextAtLevelFromPoly(levelQ, ppm.tmpB)
	tmpC := ckks2.NewCiphertextAtLevelFromPoly(levelQ-1, ppm.tmpC)

	poolDecompQP := ppm.poolDecompQP

	tmpARescale := make([][]*ckks2.Ciphertext, len(B.M))
	tmpBRescale := make([][]*ckks2.Ciphertext, len(B.M))
	for i := range B.M {
		tmpARescale[i] = make([]*ckks2.Ciphertext, innerDim-1)
		tmpBRescale[i] = make([]*ckks2.Ciphertext, innerDim-1)
		for j := 0; j < innerDim-1; j++ {
			tmpARescale[i][j] = ckks2.NewCiphertext(params, 1, levelQ-2, 0)
			tmpBRescale[i][j] = ckks2.NewCiphertextAtLevelFromPoly(levelQ-2, ppm.tmpBRescale[i][j])
		}
	}

	ciphertextAB := make([]*ckks2.Ciphertext, colsB)
	for i := range ciphertextAB {
		ciphertextAB[i] = ckks2.NewCiphertext(params, 2, levelQ, A.Scale()*B.Scale())
	}

	//Decomposes matrix B
	ciphertextA := make([]*ckks2.Ciphertext, len(B.M))
	ciphertextB := make([]*ckks2.Ciphertext, len(B.M))
	for i := range B.M {
		ciphertextA[i] = ckks2.NewCiphertext(params, 1, levelQ, 0)
		ciphertextB[i] = ckks2.NewCiphertext(params, 1, levelQ, 0)
	}
	for i := range B.M {

		ciphertextB[i] = eval.LinearTransformNew(B.M[i], mmpt.PermuteCols)[0]

		eval.GetKeySwitcher().DecomposeNTT(levelQ, levelP, alpha, ciphertextB[i].Value[1], poolDecompQP)
		for j := 0; j < innerDim-1; j++ {

			eval.MultiplyByDiagMatrix(ciphertextB[i], mmpt.RotRows[j], poolDecompQP, tmpB)

			if err := eval.Rescale(tmpB, params.Scale(), tmpBRescale[i][j]); err != nil {
				panic(err)
			}
		}

		if err := eval.Rescale(ciphertextB[i], params.Scale(), ciphertextB[i]); err != nil {
			panic(err)
		}

		ciphertextA[i] = eval.LinearTransformNew(A.M[i], mmpt.PermuteRows)[0]

		eval.GetKeySwitcher().DecomposeNTT(levelQ, levelP, alpha, ciphertextA[i].Value[1], poolDecompQP)
		for j := 0; j < innerDim-1; j++ {

			eval.MultiplyByDiagMatrix(ciphertextA[i], mmpt.RotCols[j], poolDecompQP, tmpB)

			if err := eval.Rescale(tmpB, params.Scale(), tmpARescale[i][j]); err != nil {
				panic(err)
			}
		}

		if err := eval.Rescale(ciphertextA[i], params.Scale(), ciphertextA[i]); err != nil {
			panic(err)
		}
	}

	// m1 = (A0 + A3) * (B0 + B3)
	m1 := eval.MulNew(eval.AddNew(ciphertextA[0], ciphertextA[3]), eval.AddNew(ciphertextB[0], ciphertextB[3]))
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(eval.AddNew(tmpARescale[0][v], tmpARescale[3][v]), eval.AddNew(tmpBRescale[0][v], tmpBRescale[3][v]), tmpC)
		eval.Add(m1, tmpC, m1)
	}

	// m2 = (A2 + A3) * B0
	m2 := eval.MulNew(eval.AddNew(ciphertextA[2], ciphertextA[3]), ciphertextB[0])
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(eval.AddNew(tmpARescale[2][v], tmpARescale[3][v]), tmpBRescale[0][v], tmpC)
		eval.Add(m2, tmpC, m2)
	}

	// m3 = A0 * (B1 - B3)
	m3 := eval.MulNew(ciphertextA[0], eval.SubNew(ciphertextB[1], ciphertextB[3]))
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(tmpARescale[0][v], eval.SubNew(tmpBRescale[1][v], tmpBRescale[3][v]), tmpC)
		eval.Add(m3, tmpC, m3)
	}

	// m4 = A3 * (B2 - B0)
	m4 := eval.MulNew(ciphertextA[3], eval.SubNew(ciphertextB[2], ciphertextB[0]))
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(tmpARescale[3][v], eval.SubNew(tmpBRescale[2][v], tmpBRescale[0][v]), tmpC)
		eval.Add(m4, tmpC, m4)
	}

	// m5 = (A2 + A3) * B0
	m5 := eval.MulNew(eval.AddNew(ciphertextA[0], ciphertextA[1]), ciphertextB[3])
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(eval.AddNew(tmpARescale[0][v], tmpARescale[1][v]), tmpBRescale[3][v], tmpC)
		eval.Add(m5, tmpC, m5)
	}

	// m6 = (A0 + A3) * (B0 + B3)
	m6 := eval.MulNew(eval.SubNew(ciphertextA[2], ciphertextA[0]), eval.AddNew(ciphertextB[0], ciphertextB[1]))
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(eval.SubNew(tmpARescale[2][v], tmpARescale[0][v]), eval.AddNew(tmpBRescale[0][v], tmpBRescale[1][v]), tmpC)
		eval.Add(m6, tmpC, m6)
	}

	// m7 = (A0 + A3) * (B0 + B3)
	m7 := eval.MulNew(eval.SubNew(ciphertextA[1], ciphertextA[3]), eval.AddNew(ciphertextB[2], ciphertextB[3]))
	for v := 0; v < innerDim-1; v++ {
		eval.Mul(eval.SubNew(tmpARescale[1][v], tmpARescale[3][v]), eval.AddNew(tmpBRescale[2][v], tmpBRescale[3][v]), tmpC)
		eval.Add(m7, tmpC, m7)
	}

	C.M[0] = eval.AddNew(m1, m4)
	eval.Sub(C.M[0], m5, C.M[0])
	eval.Add(C.M[0], m7, C.M[0])

	C.M[1] = eval.AddNew(m3, m5)

	C.M[2] = eval.AddNew(m2, m4)

	C.M[3] = eval.SubNew(m1, m2)
	eval.Add(C.M[3], m3, C.M[3])
	eval.Add(C.M[3], m6, C.M[3])

	eval.Relinearize(C.M[0], C.M[0])
	eval.Relinearize(C.M[1], C.M[1])
	eval.Relinearize(C.M[2], C.M[2])
	eval.Relinearize(C.M[3], C.M[3])

	if err := eval.Rescale(C.M[0], params.Scale(), C.M[0]); err != nil {
		panic(err)
	}

	if err := eval.Rescale(C.M[1], params.Scale(), C.M[1]); err != nil {
		panic(err)
	}

	if err := eval.Rescale(C.M[2], params.Scale(), C.M[2]); err != nil {
		panic(err)
	}

	if err := eval.Rescale(C.M[3], params.Scale(), C.M[3]); err != nil {
		panic(err)
	}
}

func (ppm *PackedMatrixMultiplier) MulSquareMatricesPacked(A, B *CiphertextBatchMatrix, innerDim int, C *CiphertextBatchMatrix) {

	if A.Cols() != B.Rows() {
		panic("input matrices are not compatible for multiplication")
	}

	if C.Cols() != B.Cols() || C.Rows() != A.Rows() {
		panic("output matrix is not compatible for multiplication")
	}

	level := utils2.MinInt(A.Level(), B.Level())

	str := fmt.Sprintf("0x%04o0x%04o", A.dim, level)
	mmpt, ok := ppm.multiplierMap[str]
	if !ok {
		panic(fmt.Sprintf("Multiplier [dimension:%d ; level :%d] missing", A.dim, level))
	}

	if A.Cols()*A.Rows()*B.Cols()*B.Rows() == 1 {
		// Init

		ringQ := ppm.params.RingQ()
		alpha := ppm.params.Alpha()
		eval := ppm.eval
		levelP := alpha - 1
		poolDecompQPA := eval.GetKeySwitcher().PoolDecompQP
		poolDecompQPB := ppm.poolDecompQP
		poolRescale := eval.GetKeySwitcher().PoolInvNTT

		tmpA := ckks2.NewCiphertextAtLevelFromPoly(level, ppm.tmpA)
		tmpB := ckks2.NewCiphertextAtLevelFromPoly(level, ppm.tmpB)
		tmpC := ckks2.NewCiphertextAtLevelFromPoly(level-1, ppm.tmpC)
		tmpARescale := ckks2.NewCiphertextAtLevelFromPoly(level-1, ppm.tmpARescale[0])
		tmpBRescale := ckks2.NewCiphertextAtLevelFromPoly(level-1, ppm.tmpBRescale[0][0])

		// Row & Cols permutations
		ciphertextA := eval.LinearTransformNew(A.M[0], mmpt.PermuteRows)[0]
		ciphertextB := eval.LinearTransformNew(B.M[0], mmpt.PermuteCols)[0]

		// Rescale before mul
		ringQ.DivRoundByLastModulusManyNTTLvl(level, 1, ciphertextA.Value[0], poolRescale, tmpARescale.Value[0])
		ringQ.DivRoundByLastModulusManyNTTLvl(level, 1, ciphertextA.Value[1], poolRescale, tmpARescale.Value[1])
		ringQ.DivRoundByLastModulusManyNTTLvl(level, 1, ciphertextB.Value[0], poolRescale, tmpBRescale.Value[0])
		ringQ.DivRoundByLastModulusManyNTTLvl(level, 1, ciphertextB.Value[1], poolRescale, tmpBRescale.Value[1])

		scaleTmp := ciphertextA.Scale / ppm.params.QiFloat64(ciphertextA.Level())
		tmpARescale.Scale, tmpBRescale.Scale = scaleTmp, scaleTmp

		// First element of the inner product (without relinearization)
		C.M[0] = eval.MulNew(tmpARescale, tmpBRescale)

		// Decompose A and B for hoisting linear transforms
		eval.GetKeySwitcher().DecomposeNTT(ciphertextA.Level(), levelP, alpha, ciphertextA.Value[1], poolDecompQPA)
		eval.GetKeySwitcher().DecomposeNTT(ciphertextB.Level(), levelP, alpha, ciphertextB.Value[1], poolDecompQPB)

		// Reset pool to the correct level
		tmpARescale = ckks2.NewCiphertextAtLevelFromPoly(ciphertextA.Level()-2, ppm.tmpARescale[0])
		tmpBRescale = ckks2.NewCiphertextAtLevelFromPoly(ciphertextA.Level()-2, ppm.tmpBRescale[0][0])
		tmpARescale.Scale, tmpBRescale.Scale = scaleTmp, scaleTmp

		// Inner product
		for i := 0; i < innerDim-1; i++ {

			// Row & Cols rotations
			eval.MultiplyByDiagMatrix(ciphertextA, mmpt.RotCols[i], poolDecompQPA, tmpA)
			ringQ.DivRoundByLastModulusManyNTTLvl(tmpA.Level(), 2, tmpA.Value[0], poolRescale, tmpARescale.Value[0])
			ringQ.DivRoundByLastModulusManyNTTLvl(tmpA.Level(), 2, tmpA.Value[1], poolRescale, tmpARescale.Value[1])

			if len(mmpt.RotRows[i].Vec) == 1 {
				eval.PermuteNTTHoisted(tmpB.Level(), ciphertextB.Value[0], ciphertextB.Value[1], poolDecompQPB, (i+1)*innerDim, tmpB.Value[0], tmpB.Value[1])
				ringQ.MulScalarLvl(tmpB.Level()+1, tmpB.Value[0], ringQ.Modulus[tmpB.Level()], tmpB.Value[0])
				ringQ.MulScalarLvl(tmpB.Level()+1, tmpB.Value[1], ringQ.Modulus[tmpB.Level()], tmpB.Value[1])
			} else {
				eval.MultiplyByDiagMatrix(ciphertextB, mmpt.RotRows[i], poolDecompQPB, tmpB)
			}

			ringQ.DivRoundByLastModulusManyNTTLvl(tmpB.Level(), 2, tmpB.Value[0], poolRescale, tmpBRescale.Value[0])
			ringQ.DivRoundByLastModulusManyNTTLvl(tmpB.Level(), 2, tmpB.Value[1], poolRescale, tmpBRescale.Value[1])

			// Mul without relinearization
			eval.Mul(tmpARescale, tmpBRescale, tmpC)

			// Add
			eval.Add(C.M[0], tmpC, C.M[0])
		}

		// Relinearize and rescale
		eval.Relinearize(C.M[0], C.M[0])
		if err := eval.Rescale(C.M[0], ppm.params.Scale(), C.M[0]); err != nil {
			panic(err)
		}

	} else {

		params := ppm.params
		eval := ppm.eval
		levelQ := utils2.MinInt(A.Level(), B.Level())
		levelP := params.PCount() - 1
		alpha := params.PCount()

		rowsA := A.Rows()
		colsA := A.Cols()
		colsB := B.Cols()

		tmpA := ckks2.NewCiphertextAtLevelFromPoly(levelQ, ppm.tmpA)
		tmpB := ckks2.NewCiphertextAtLevelFromPoly(levelQ, ppm.tmpB)
		tmpC := ckks2.NewCiphertextAtLevelFromPoly(levelQ-1, ppm.tmpC)

		poolDecompQP := ppm.poolDecompQP

		tmpARescale := make([]*ckks2.Ciphertext, innerDim-1)
		for i := 0; i < innerDim-1; i++ {
			tmpARescale[i] = ckks2.NewCiphertextAtLevelFromPoly(levelQ-2, ppm.tmpARescale[i])
		}

		tmpBRescale := make([][]*ckks2.Ciphertext, len(B.M))
		for i := range B.M {
			tmpBRescale[i] = make([]*ckks2.Ciphertext, innerDim-1)
			for j := 0; j < innerDim-1; j++ {
				tmpBRescale[i][j] = ckks2.NewCiphertextAtLevelFromPoly(levelQ-2, ppm.tmpBRescale[i][j])
			}
		}

		ciphertextAB := make([]*ckks2.Ciphertext, colsB)
		for i := range ciphertextAB {
			ciphertextAB[i] = ckks2.NewCiphertext(params, 2, levelQ, A.Scale()*B.Scale())
		}

		//Decomposes matrix B
		ciphertextB := make([]*ckks2.Ciphertext, len(B.M))
		for i := range B.M {
			ciphertextB[i] = ckks2.NewCiphertext(params, 1, levelQ, 0)
		}
		for i := range B.M {

			ciphertextB[i] = eval.LinearTransformNew(B.M[i], mmpt.PermuteCols)[0]

			eval.GetKeySwitcher().DecomposeNTT(levelQ, levelP, alpha, ciphertextB[i].Value[1], poolDecompQP)
			for j := 0; j < innerDim-1; j++ {

				eval.MultiplyByDiagMatrix(ciphertextB[i], mmpt.RotRows[j], poolDecompQP, tmpB)

				if err := eval.Rescale(tmpB, params.Scale(), tmpBRescale[i][j]); err != nil {
					panic(err)
				}
			}

			if err := eval.Rescale(ciphertextB[i], params.Scale(), ciphertextB[i]); err != nil {
				panic(err)
			}
		}

		// Processes each rows of A
		for i := 0; i < rowsA; i++ {

			// Processes each element of a rows of A
			for j := 0; j < colsA; j++ {

				// Linear transforms the coefficient j of the row i of A
				ciphertextA := eval.LinearTransformNew(A.M[i*colsA+j], mmpt.PermuteRows)[0]

				// Decompose the coefficient j of the row i of A
				eval.GetKeySwitcher().DecomposeNTT(levelQ, levelP, alpha, ciphertextA.Value[1], poolDecompQP)

				for k := 0; k < innerDim-1; k++ {
					eval.MultiplyByDiagMatrix(ciphertextA, mmpt.RotCols[k], poolDecompQP, tmpA)

					if err := eval.Rescale(tmpA, params.Scale(), tmpARescale[k]); err != nil {
						panic(err)
					}
				}

				if err := eval.Rescale(ciphertextA, params.Scale(), ciphertextA); err != nil {
					panic(err)
				}

				// Iterates over the coefficients of the column j of B and sums the result on the accumulator
				for k := 0; k < colsB; k++ {
					if j == 0 {
						eval.Mul(ciphertextA, ciphertextB[k+j*colsB], ciphertextAB[k])
					} else {
						if err := eval.Rescale(ciphertextA, params.Scale(), ciphertextA); err != nil {
							panic(err)
						}
						eval.Mul(ciphertextA, ciphertextB[k+j*colsB], tmpC)
						eval.Add(ciphertextAB[k], tmpC, ciphertextAB[k])
					}

					for v := 0; v < innerDim-1; v++ {
						eval.Mul(tmpARescale[v], tmpBRescale[k+j*colsB][v], tmpC)
						eval.Add(ciphertextAB[k], tmpC, ciphertextAB[k])
					}
				}
			}

			// Once a row of A has been processed, relinearize the sum and rescales
			for j := 0; j < colsB; j++ {
				eval.Relinearize(ciphertextAB[j], C.M[i*colsB+j])
				if err := eval.Rescale(C.M[i*colsB+j], params.Scale(), C.M[i*colsB+j]); err != nil {
					panic(err)
				}
			}
		}
	}
}

func (ppm *PackedMatrixMultiplier) Transpose(A *CiphertextBatchMatrix, B *CiphertextBatchMatrix) {
	rows := A.Rows()
	cols := A.Cols()
	scale := A.Scale()

	if len(A.M) != len(B.M) {
		panic("output matrix is not compatible with input matrix")
	}

	level := utils2.MinInt(A.Level(), B.Level())

	str := fmt.Sprintf("0x%04o0x%04o", A.dim, level)
	transposeLT, ok := ppm.transposerMap[str]
	if !ok {
		panic(fmt.Sprintf("Transpose [dimension:%d ; level :%d] missing", A.dim, level))
	}

	for i := range A.M {
		B.M[i] = ppm.eval.LinearTransformNew(A.M[i], transposeLT.PtDiagMatrix)[0]
		if err := ppm.eval.Rescale(B.M[i], scale, B.M[i]); err != nil {
			panic(err)
		}
	}

	tmp := make([]*ckks2.Ciphertext, A.Rows()*A.Cols())
	for i := 0; i < A.Rows(); i++ {
		for j := 0; j < A.Cols(); j++ {
			tmp[j*rows+i] = B.M[i*cols+j]
		}
	}
	B.M = tmp
	B.rows, B.cols = cols, rows
}

func (ppm *PackedMatrixMultiplier) MulPlainLeft(A []*PlaintextBatchMatrix, B *CiphertextBatchMatrix, dimmension int, C []*CiphertextBatchMatrix) {

	if A[0].Dim() != B.Dim() || A[0].Dim() != C[0].Dim() {
		panic("input/output matrices do not share the same inner dimension")
	}

	if A[0].Cols() != B.Rows() {
		panic("input matrices are not compatible for multiplication")
	}

	if C[0].Cols() != B.Cols() || C[0].Rows() != A[0].Rows() {
		panic("output matrix is not compatible for multiplication")
	}

	eval := ppm.eval
	params := ppm.params
	acc := ckks2.NewCiphertext(params, 1, B.M[0].Level(), params.Scale())

	level := utils2.MinInt(A[0].Level(), B.Level())

	str := fmt.Sprintf("0x%04o0x%04o", A[0].dim, level)
	mmpt, ok := ppm.multiplierMap[str]
	if !ok {
		panic(fmt.Sprintf("Multiplier [dimension:%d ; level :%d] missing", A[0].Dim(), level))
	}

	rowsA := A[0].rows
	rowsB := B.rows
	colsB := B.cols

	// Loops over each element of W0
	for i := 0; i < rowsB; i++ {

		for j := 0; j < colsB; j++ {

			indexB := i*colsB + j

			//fmt.Printf("Rot(B[%d])\n", indexB)

			W0ijRotated := eval.LinearTransformNew(B.M[indexB], mmpt.RotRows)

			for k := 0; k < rowsA; k++ {

				indexA := k*rowsB + i

				indexC := k*colsB + j

				//fmt.Printf("A[%d] x B[%d] -> C[%d]\n", indexA, indexB, indexC)

				for idxCt, Ai := range A {
					for u := 0; u < dimmension; u++ {

						if u == 0 {
							ppm.eval.Mul(B.M[indexB], Ai.M[indexA][u], acc)
							ppm.eval.Add(C[idxCt].M[indexC], acc, C[idxCt].M[indexC])
						} else {
							ppm.eval.Mul(W0ijRotated[u-1], Ai.M[indexA][u], acc)
							ppm.eval.Add(C[idxCt].M[indexC], acc, C[idxCt].M[indexC])
						}
					}
				}
			}
		}
	}

	for idxCt := range A {
		for i := range C[idxCt].M {
			ppm.eval.Rescale(C[idxCt].M[i], ppm.params.Scale(), C[idxCt].M[i])
		}
	}
}

func (ppm *PackedMatrixMultiplier) Add(A, B, C *CiphertextBatchMatrix) {
	if len(A.M) != len(B.M) {
		panic("input matrices are not compatible for addition")
	}

	if len(A.M) != len(C.M) {
		panic("output matrix is not compatible with input matrices")
	}

	for i := range A.M {
		ppm.eval.Add(A.M[i], B.M[i], C.M[i])
	}
}

func (ppm *PackedMatrixMultiplier) AddPlain(A *CiphertextBatchMatrix, B *PlaintextBatchMatrix, C *CiphertextBatchMatrix) {
	if len(A.M) != len(B.M) {
		panic("input matrices are not compatible for addition")
	}

	if len(A.M) != len(C.M) {
		panic("output matrix is not compatible with input matrices")
	}

	for i := range A.M {
		ppm.eval.Add(A.M[i], B.M[i][0], C.M[i])
	}
}

func (ppm *PackedMatrixMultiplier) Sub(A, B, C *CiphertextBatchMatrix) {
	if len(A.M) != len(B.M) {
		panic("input matrices are not compatible for subtraction")
	}

	if len(A.M) != len(C.M) {
		panic("output matrix is not compatible with input matrices")
	}

	for i := range A.M {
		ppm.eval.Sub(A.M[i], B.M[i], C.M[i])
	}
}

func (ppm *PackedMatrixMultiplier) Dot(A, B, C *CiphertextBatchMatrix) {
	if A.Rows() != B.Rows() || A.Cols() != B.Cols() {
		panic("input matrices are not compatible for dot mul")
	}

	if C.Rows() != A.Rows() || C.Cols() != A.Cols() {
		panic("output matrix is not compatible for dot mul")
	}

	for i := range A.M {
		ppm.eval.MulRelin(A.M[i], B.M[i], C.M[i])
		if err := ppm.eval.Rescale(C.M[i], ppm.params.Scale(), C.M[i]); err != nil {
			panic(err)
		}
	}
}

func (ppm *PackedMatrixMultiplier) EvalPoly(A *CiphertextBatchMatrix, poly *ckks2.Polynomial) (B *CiphertextBatchMatrix) {

	var err error

	B = NewCiphertextBatchMatrix(A.Rows(), A.Cols(), A.dim, make([]*ckks2.Ciphertext, A.Rows()*A.Cols()))

	for i := range A.M {
		if B.M[i], err = ppm.eval.EvaluatePoly(A.M[i], poly, A.Scale()); err != nil {
			panic(err)
		}
	}

	return
}

func (ppm *PackedMatrixMultiplier) InnerSumLog(A *CiphertextBatchMatrix, batch, n int, ctOut *CiphertextBatchMatrix) {

	for i := range A.M {
		ppm.eval.InnerSumLog(A.M[i], batch, n, ctOut.M[i])
	}
}

func (ppm *PackedMatrixMultiplier) Rescale(A *CiphertextBatchMatrix, scale float64, B *CiphertextBatchMatrix) {
	for i := range A.M {
		if err := ppm.eval.Rescale(A.M[i], scale, A.M[i]); err != nil {
			panic(err)
		}
	}
}
