package cipherUtils

import (
	"errors"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
)

/*
	Format weights in diagonal form for multiplication algorithm
	Additionally use the complex trick to effectively divide by half the
	size of the matrix to be multiplied
	refer to page 3: https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
*/
func FormatWeights(w [][]float64, leftdim int) (m [][]complex128) {

	scaling := complex(0.5, 0)

	m = make([][]complex128, (len(w)+1)/2)

	for i := 0; i < len(w)>>1; i++ {

		m[i] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {

			cReal := w[(i*2+0+j)%len(w)][j]
			cImag := w[(i*2+1+j)%len(w)][j]

			for k := 0; k < leftdim; k++ {
				m[i][j*leftdim+k] = scaling * complex(cReal, -cImag) // 0.5 factor for imaginary part cleaning: (a+bi) + (a-bi) = 2a
			}
		}
	}
	//odd
	if len(w)&1 == 1 {

		idx := len(m) - 1

		m[idx] = make([]complex128, leftdim*len(w[0]))

		for j := 0; j < len(w[0]); j++ {
			cReal := w[(idx*2+j)%len(w)][j]
			for k := 0; k < leftdim; k++ {
				m[idx][j*leftdim+k] = scaling * complex(cReal, 0)
			}
		}
	}

	return
}

//Returns matrix in diagonal form as map. W has to be square
func FormatWeightsAsMap(W [][]float64, leftdim int, slots int, complexTrick bool) (map[int][]complex128, error) {
	if !complexTrick {
		d := len(W)
		if d != len(W[0]) {
			return nil, errors.New("Non square")
		}
		if d*leftdim*2 > slots {
			return nil, errors.New("d * leftdim * 2 > slots")
		}
		nonZeroDiags := make(map[int][]complex128) //rotation -> diag to be multiplied by
		for i := 0; i < d; i++ {
			isZero := true
			diag := make([]complex128, d*leftdim)
			z := 0
			for j := 0; j < d; j++ {
				for k := 0; k < leftdim; k++ {
					diag[z] = complex(W[(j+i)%d][(j)%d], 0)
					z++
				}
				if diag[j] != 0 {
					isZero = false
				}
			}
			if !isZero {
				nonZeroDiags[leftdim*i] = plainUtils.ReplicateComplexArray(diag, 2)
			}
		}
		return nonZeroDiags, nil
	} else {
		d := len(W)
		if d != len(W[0]) {
			return nil, errors.New("Non square")
		}
		if d*leftdim*2 > slots {
			return nil, errors.New("d * rowIn * 2 > slots")
		}
		scaling := complex(0.5, 0)

		nonZeroDiags := make(map[int][]complex128) //rotation -> diag to be multiplied by
		m := make([][]complex128, (len(W)+1)/2)

		for i := 0; i < len(W)>>1; i++ {

			m[i] = make([]complex128, leftdim*len(W[0]))

			for j := 0; j < len(W[0]); j++ {

				cReal := W[(i*2+0+j)%len(W)][j]
				cImag := W[(i*2+1+j)%len(W)][j]

				for k := 0; k < leftdim; k++ {
					m[i][j*leftdim+k] = scaling * complex(cReal, -cImag) // 0.5 factor for imaginary part cleaning: (a+bi) + (a-bi) = 2a
				}
			}
		}
		//odd
		if len(W)&1 == 1 {

			idx := len(m) - 1

			m[idx] = make([]complex128, leftdim*len(W[0]))

			for j := 0; j < len(W[0]); j++ {
				cReal := W[(idx*2+j)%len(W)][j]
				for k := 0; k < leftdim; k++ {
					m[idx][j*leftdim+k] = scaling * complex(cReal, 0)
				}
			}
		}
		for i := range m {
			nonZeroDiags[2*i*leftdim] = plainUtils.ReplicateComplexArray(m[i], 2)
		}
		return nonZeroDiags, nil
	}
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
