package cipherUtils

import (
	"fmt"
	"math"
)

func FormatWeights(w [][]float64, leftdim int) (m [][]complex128) {
	/*
		Format weights in diagonal form for multiplication algorithm
		Additionally use the complex trick to effectively divide by half the
		size of the matrix to be multiplied
		refer to page 3: https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
	*/

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

func FormatWeightsAsMap(w [][]float64, leftdim int) (m map[int][]float64, nonZeroDiags []int) {
	/*
		Format weights in diagonal form for multiplication algorithm
		Additionally use the complex trick to effectively divide by half the
		size of the matrix to be multiplied
		refer to page 3: https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
	*/
	m = make(map[int][]float64)

	//complex pack
	wRow := len(w)
	wCol := len(w[0])
	wPacked := make([][]complex128, int(math.Ceil(float64(wRow)/2)))
	ip := 0
	for i := 0; i < wCol-1; i++ {
		wPacked[ip] = make([]complex128, wCol)
		if i%2 == 0 {
			if i+1 < wCol {
				for j := 0; j < wCol; j++ {
					wPacked[ip][j] = 0.5 * complex(w[i][j], -1*w[i+1][j])
				}
			} else {
				for j := 0; j < wCol; j++ {
					wPacked[ip][j] = 0.5 * complex(w[i][j], 0)
				}
			}
			ip++
		}

	}
	for i := range wPacked {
		fmt.Println(i)
		fmt.Println(wPacked[i])
	}
	nonZeroDiags = make([]int, len(wPacked))
	for i := 0; i < len(wPacked); i++ {
		d := make([]float64, leftdim*len(w[0]))
		nonZeroDiags[i] = i
		for j := 0; j < wCol; j++ {

			cReal := w[(i+j)%len(w)][j]
			for k := 0; k < leftdim; k++ {
				d[j*leftdim+k] = cReal
			}
		}
		m[i] = d
	}
	return
}

func FormatInput(w [][]float64) (v []float64) {
	v = make([]float64, len(w)*len(w[0])) //missing a 2

	for i := 0; i < len(w[0]); i++ {
		for j := 0; j < len(w); j++ {
			v[i*len(w)+j] = w[j][i] //transposed
		}
	}

	return
}
