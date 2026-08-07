//go:build amd64

package gguf

//go:noescape
func q6KCoeffDotAsm(q8 *[256]int8, coeff *[256]int16) int32

func q6KCoeffDot(q8 *[256]int8, coeff *[256]int16) int32 { return q6KCoeffDotAsm(q8, coeff) }
