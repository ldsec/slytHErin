package cipherUtils

import "github.com/tuneinsight/lattigo/v3/utils"

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

func FormatWeightsAsMap(w [][]float64, leftdim int, complexTrick bool) (map[int][]complex128, []int) {
	/*
		Format weights in diagonal form for multiplication algorithm
		Additionally use the complex trick to effectively divide by half the
		size of the matrix to be multiplied
		refer to page 3: https://www.biorxiv.org/content/biorxiv/early/2022/01/11/2022.01.10.475610/DC1/embed/media-1.pdf?download=true
	*/
	if complexTrick {
		scaling := complex(0.5, 0)

		m := make([][]complex128, (len(w)+1)/2)

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

		mp := map[int][]complex128{}
		nonZeroDiags := make([]int, len(m))

		for i, d := range m {
			nonZeroDiags[i] = i
			mp[i] = d
		}
		return mp, nonZeroDiags
	} else {
		dim := len(w[0])
		mDiag := make(map[int][]complex128)
		for i := 0; i < dim; i++ {

			tmp := make([]complex128, dim)

			for j := 0; j < dim-i; j++ {

				tmp[j] = complex(w[j][(j+i)%dim], 0)
			}
		}

		for i := 1; i < len(w); i++ {
			tmp := make([]complex128, dim)
			for j := 0; j < dim-i; j++ {
				tmp[j] = complex(w[i+j][j], 0)
			}

			mDiag[-i] = utils.RotateComplex128Slice(tmp, -i)

		}

		return mDiag, nil
	}
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
