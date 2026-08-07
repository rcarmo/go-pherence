//go:build !amd64

package gguf

func q6KCoeffDot(q8 *[256]int8, coeff *[256]int16) int32 {
	var sum int32
	for i := range q8 {
		sum += int32(q8[i]) * int32(coeff[i])
	}
	return sum
}

func q6KCoeffDot8(q8 *[256]int8, coeff *[256]int16, out *[8]int32) {
	for i := range q8 {
		out[(i/2)%8] += int32(q8[i]) * int32(coeff[i])
	}
}
