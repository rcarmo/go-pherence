//go:build amd64

package gguf

//go:noescape
func q6KCoeffDotAsm(q8 *[256]int8, coeff *[256]int16) int32

//go:noescape
func q6KExpandCoeffAsm(block *[210]byte, coeff *[256]int16)

//go:noescape
func q6KCoeffDot8Asm(q8 *[256]int8, coeff *[256]int16, out *[8]int32)

func q6KCoeffDot(q8 *[256]int8, coeff *[256]int16) int32 { return q6KCoeffDotAsm(q8, coeff) }

func q6KExpandCoeff(block *[210]byte, coeff *[256]int16) {
	q6KExpandCoeffAsm(block, coeff)
}

func q6KCoeffDot8(q8 *[256]int8, coeff *[256]int16, out *[8]int32) {
	q6KCoeffDot8Asm(q8, coeff, out)
}
