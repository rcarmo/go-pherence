//go:build !amd64

package gguf

func q6KExpandCoeff(block *[210]byte, coeff *[256]int16) {
	ql, qh, scales := block[:128], block[128:192], block[192:208]
	for halfBlock := 0; halfBlock < 2; halfBlock++ {
		qlOff, qhOff, base := halfBlock*64, halfBlock*32, halfBlock*128
		for l := 0; l < 32; l++ {
			is := l / 16
			coeff[base+l] = int16(int8(scales[halfBlock*8+is])) * (int16((ql[qlOff+l]&15)|(((qh[qhOff+l]>>0)&3)<<4)) - 32)
			coeff[base+l+32] = int16(int8(scales[halfBlock*8+is+2])) * (int16((ql[qlOff+l+32]&15)|(((qh[qhOff+l]>>2)&3)<<4)) - 32)
			coeff[base+l+64] = int16(int8(scales[halfBlock*8+is+4])) * (int16((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32)
			coeff[base+l+96] = int16(int8(scales[halfBlock*8+is+6])) * (int16((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32)
		}
	}
}

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

func q6KBlockDot(block *[210]byte, q8 *[256]int8) int32 {
	var coeff [256]int16
	q6KExpandCoeff(block, &coeff)
	return q6KCoeffDot(q8, &coeff)
}

func dotQ6KQ8KGemvVNNI(raw []byte, y []q8KBlock, blocks int) (float32, bool) {
	return 0, false
}

func q6KBlockDotAVX2(block *[210]byte, q8 *[256]int8) int32 {
	return q6KBlockDot(block, q8)
}

func q6KBlockDotVNNI(block *[210]byte, q8 *[256]int8) (int32, bool) {
	return 0, false
}
