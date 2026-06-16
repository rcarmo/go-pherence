//go:build amd64

package diffusiongemma

const hasQ6KBlockCoeffISumSIMD = true

//go:noescape
func q6KBlockCoeffISumAsm(q8 []int8, coeff *[256]int16) int32

func q6KBlockCoeffISumFast(q8 []int8, coeff *[256]int16) int32 {
	return q6KBlockCoeffISumAsm(q8, coeff)
}
