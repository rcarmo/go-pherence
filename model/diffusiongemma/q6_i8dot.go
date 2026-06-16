package diffusiongemma

func q6KBlockISumScalar(q8 []int8, q6 *[256]int8, scales []byte) (int32, bool) {
	if len(q8) < 256 || q6 == nil || len(scales) < 16 {
		return 0, false
	}
	var isum int32
	for group := 0; group < 16; group++ {
		scale := int32(int8(scales[group]))
		base := group * 16
		acc := int32(0)
		for i := 0; i < 16; i++ {
			acc += int32(q8[base+i]) * int32(q6[base+i])
		}
		isum += scale * acc
	}
	return isum, true
}

func q6KBlockISum(q8 []int8, q6 *[256]int8, scales []byte) (int32, bool) {
	return q6KBlockISumScalar(q8, q6, scales)
}

func q6KBlockScaledCoeffs(q6 *[256]int8, scales []byte) ([256]int16, bool) {
	var coeff [256]int16
	if q6 == nil || len(scales) < 16 {
		return coeff, false
	}
	for group := 0; group < 16; group++ {
		scale := int16(int8(scales[group]))
		base := group * 16
		for i := 0; i < 16; i++ {
			coeff[base+i] = scale * int16(q6[base+i])
		}
	}
	return coeff, true
}

func q6KBlockCoeffISum(q8 []int8, coeff *[256]int16) (int32, bool) {
	if len(q8) < 256 || coeff == nil {
		return 0, false
	}
	if hasQ6KBlockCoeffISumSIMD {
		return q6KBlockCoeffISumFast(q8[:256], coeff), true
	}
	return q6KBlockCoeffISumScalar(q8[:256], coeff), true
}

func q6KBlockCoeffISumScalar(q8 []int8, coeff *[256]int16) int32 {
	var sum int32
	for i := 0; i < 256; i++ {
		sum += int32(q8[i]) * int32(coeff[i])
	}
	return sum
}
