package utils

import "math"

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0.0
	}
}

func SoftReLu(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}
