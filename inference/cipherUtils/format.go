package cipherUtils

import "fmt"

func FormatWeights(w [][]float64, leftdim int) (m [][]complex128) {
	/*
		Format weights in diagonal form for Halevi-Shoup multiplication algorithm
		Additionally use the complex trick to effectively divide by half the
		size of the matrix to be multiplied
		refer to page 3: https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
	*/
	m = make([][]complex128, (len(w)+1)/2)
	for i := 0; i < len(w)>>1; i++ {

		m[i] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {

			cReal := w[(i*2+0+j)%len(w)][j]
			fmt.Println("cReal", cReal)
			cImag := w[(i*2+1+j)%len(w)][j]
			fmt.Println("cImag", cImag)

			for k := 0; k < leftdim; k++ {
				//fmt.Printf("m value at place %d,%d = 0.5(%f, %f.i)\n", i, j*leftdim+k, cReal, -cImag)
				m[i][j*leftdim+k] = 0.5 * complex(cReal, -cImag) // 0.5 factor for imaginary part cleaning: (a+bi) + (a-bi) = 2a
			}
		}
	}

	if len(w)&1 == 1 {
		//if odd
		idx := len(m) - 1

		m[idx] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {
			cReal := w[(idx*2+j)%len(w)][j]
			for k := 0; k < leftdim; k++ {
				//fmt.Printf("m value at place %d,%d = 0.5(%f, %f.i)\n", idx, j*leftdim+k, cReal, 0.0)
				m[idx][j*leftdim+k] = 0.5 * complex(cReal, 0)
			}
		}
	}

	return
}

func FormatInput(w [][]float64) (v []float64) {
	/*
		Prepare input for multiplication following the Halevi-Shoup method
		with diagonal-packed weights,
		i.e computes Flatten(w.T) + 0s padding
		if w is nxm --> v is 2xnxm
	*/
	v = make([]float64, len(w)*len(w[0])*2)

	for i := 0; i < len(w[0]); i++ {
		for j := 0; j < len(w); j++ {
			v[i*len(w)+j] = w[j][i]
		}
	}
	//fmt.Println("Input:", w)
	//fmt.Println("Formatted Input:", v)
	return v
}
