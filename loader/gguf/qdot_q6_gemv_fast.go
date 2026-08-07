package gguf

import (
	"encoding/binary"

	"github.com/rcarmo/go-pherence/half"
)

func dotQ6KQ8KGemvFast(raw []byte, y []q8KBlock, blocks int) float32 {
	var sum float32
	for bi := 0; bi < blocks; bi++ {
		blk := raw[bi*210 : (bi+1)*210]
		ql, qh, scales := blk[:128], blk[128:192], blk[192:208]
		var coeff [256]int16
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
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:])) * y[bi].d
		sum += d * float32(q6KCoeffDot(&y[bi].qs, &coeff))
	}
	return sum
}
