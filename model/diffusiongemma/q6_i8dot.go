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
