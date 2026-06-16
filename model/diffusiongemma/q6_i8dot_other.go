//go:build !amd64

package diffusiongemma

const hasQ6KBlockCoeffISumSIMD = false

func q6KBlockCoeffISumFast(q8 []int8, coeff *[256]int16) int32 { return 0 }
