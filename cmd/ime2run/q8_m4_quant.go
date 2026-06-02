package main

import "math"

// quantizeQ8RowsM4Bytes builds the native M4 A layout directly:
// per K32 block: fp32 scale[4], int16 negative_sum[4], int8 q[4][32].
// This mirrors llama.cpp quantize_a_4row_i8 for blk_len=32.
func quantizeQ8RowsM4Bytes(acts [4][]float32, kBlks int) []byte {
	if q8RoundOn {
		var rows [4][]byte
		for i := 0; i < 4; i++ {
			rows[i] = quantizeQ8Blocks32Bytes(acts[i])
		}
		return packQ8RowsM4(rows, kBlks)
	}
	out := make([]byte, kBlks*k3I8I4ABlockM4Bytes)
	for sb := 0; sb < kBlks; sb++ {
		dst := sb * k3I8I4ABlockM4Bytes
		base := sb * 32
		for r := 0; r < 4; r++ {
			var maxAbs float32
			row := acts[r]
			for i := 0; i < 32; i++ {
				a := row[base+i]
				if a < 0 {
					a = -a
				}
				if a > maxAbs {
					maxAbs = a
				}
			}
			if maxAbs == 0 {
				continue
			}
			scale := maxAbs / 127.0
			bits := math.Float32bits(scale)
			out[dst+r*4+0] = byte(bits)
			out[dst+r*4+1] = byte(bits >> 8)
			out[dst+r*4+2] = byte(bits >> 16)
			out[dst+r*4+3] = byte(bits >> 24)
			rep := float32(127.0) / maxAbs
			var sum int32
			for i := 0; i < 32; i++ {
				v := row[base+i] * rep
				if q8RoundOn {
					v = float32(math.RoundToEven(float64(v)))
				}
				if v > 127 {
					v = 127
				} else if v < -128 {
					v = -128
				}
				q := int8(v)
				out[dst+24+r*32+i] = byte(q)
				sum += int32(q)
			}
			sumNeg := int16(-sum)
			out[dst+16+r*2+0] = byte(sumNeg)
			out[dst+16+r*2+1] = byte(uint16(sumNeg) >> 8)
		}
	}
	return out
}
