package cipherUtils

/*
	Format weights in diagonal form for multiplication algorithm
	Additionally use the complex trick to effectively divide by half the
	size of the matrix to be multiplied
	refer to page 3: https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
*/
func FormatWeights(w [][]float64, leftdim int) (m map[int][]complex128) {

	scaling := complex(0.5, 0)

	m = make(map[int][]complex128)

	for i := 0; i < len(w)>>1; i++ {

		//m[i] = make([]complex128, leftdim*len(w[0]))
		d := make([]complex128, leftdim*len(w[0]))
		isZero := true
		for j := 0; j < len(w[0]); j++ {

			cReal := w[(i*2+0+j)%len(w)][j]
			cImag := w[(i*2+1+j)%len(w)][j]
			if cReal != 0 || cImag != 0 {
				isZero = false
			}
			for k := 0; k < leftdim; k++ {
				d[j*leftdim+k] = scaling * complex(cReal, -cImag) // 0.5 factor for imaginary part cleaning: (a+bi) + (a-bi) = 2a
			}
		}
		if !isZero {
			m[2*i*leftdim] = d
		}
	}
	//odd
	if len(w)&1 == 1 {

		idx := (len(w)+1)/2 - 1

		d := make([]complex128, leftdim*len(w[0]))
		isZero := true
		for j := 0; j < len(w[0]); j++ {
			cReal := w[(idx*2+j)%len(w)][j]
			if cReal != 0 {
				isZero = false
			}
			for k := 0; k < leftdim; k++ {
				d[j*leftdim+k] = scaling * complex(cReal, 0)
			}
		}
		if !isZero {
			m[2*idx*leftdim] = d
		}
	}

	return
}

//Transposes and flattens input
func FormatInput(w [][]float64) (v []float64) {
	v = make([]float64, len(w)*len(w[0])) //missing a 2
	for i := 0; i < len(w[0]); i++ {
		for j := 0; j < len(w); j++ {
			v[i*len(w)+j] = w[j][i] //transposed
		}
	}

	return
}
