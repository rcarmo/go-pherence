package main

import (
	"math"
	"unsafe"
)

const q6KBlockBytes = 210

func q6kD(raw []byte, off int) float32 {
	bits := uint16(raw[off+208]) | uint16(raw[off+209])<<8
	return fp16ToFloat(bits)
}

// repackQ6KRawToQ80x32 ports llama.cpp repack_q6_k_to_q8_0_32_bl_ref.
// It converts raw Q6_K rows into the native q8_0_32x32 B layout consumed by
// gemm_kernel_i8i8_m{1,4}: fp16 d[32] + int8 qs[32][32] per K32/N32 tile.
func repackQ6KRawToQ80x32(M, K int, raw []byte) q8Q80x32 {
	if M%32 != 0 || K%256 != 0 || len(raw) < M*(K/256)*q6KBlockBytes {
		return q8Q80x32{}
	}
	nblocks := K / 256
	subs := K / 32
	groups := M / 32
	out := q8Q80x32{M: M, K: K, BData: make([]byte, groups*subs*(64+1024)), Valid: true}
	var aux [32]int8
	for rg := 0; rg < groups; rg++ {
		for x := 0; x < nblocks; x++ {
			for bi := 0; bi < 8; bi++ {
				sb := x*8 + bi
				base := (rg*subs + sb) * 1088
				for r := 0; r < 32; r++ {
					row := rg*32 + r
					blkOff := (row*nblocks + x) * q6KBlockBytes
					q4 := raw[blkOff : blkOff+128]
					qh := raw[blkOff+128 : blkOff+192]
					scales := *(*[16]int8)(unsafe.Pointer(&raw[blkOff+192]))
					d := q6kD(raw, blkOff)
					q4Base := 64 * (bi / 4)
					qhBase := 32 * (bi / 4)
					switch bi % 4 {
					case 0:
						for l := 0; l < 32; l++ {
							aux[l] = int8((q4[q4Base+l]&0x0f)|(((qh[qhBase+l]>>0)&3)<<4)) - 32
						}
					case 1:
						for l := 0; l < 32; l++ {
							aux[l] = int8((q4[q4Base+l+32]&0x0f)|(((qh[qhBase+l]>>2)&3)<<4)) - 32
						}
					case 2:
						for l := 0; l < 32; l++ {
							aux[l] = int8((q4[q4Base+l]>>4)|(((qh[qhBase+l]>>4)&3)<<4)) - 32
						}
					case 3:
						for l := 0; l < 32; l++ {
							aux[l] = int8((q4[q4Base+l+32]>>4)|(((qh[qhBase+l]>>6)&3)<<4)) - 32
						}
					}
					scale0 := float32(scales[bi*2+0]) * d
					scale1 := float32(scales[bi*2+1]) * d
					var maxAbs float32
					for l := 0; l < 16; l++ {
						v := float32(aux[l]) * scale0
						if v < 0 { v = -v }
						if v > maxAbs { maxAbs = v }
					}
					for l := 16; l < 32; l++ {
						v := float32(aux[l]) * scale1
						if v < 0 { v = -v }
						if v > maxAbs { maxAbs = v }
					}
					reflectScale := float32(0)
					if maxAbs != 0 {
						reflectScale = maxAbs / 127.0
					}
					if reflectScale != 0 {
						rs0 := scale0 / reflectScale
						rs1 := scale1 / reflectScale
						for l := 0; l < 16; l++ {
							q := float32(math.Round(float64(float32(aux[l]) * rs0)))
							if q > 127 { q = 127 } else if q < -128 { q = -128 }
							aux[l] = int8(q)
						}
						for l := 16; l < 32; l++ {
							q := float32(math.Round(float64(float32(aux[l]) * rs1)))
							if q > 127 { q = 127 } else if q < -128 { q = -128 }
							aux[l] = int8(q)
						}
					} else {
						for l := range aux { aux[l] = 0 }
					}
					bits := f32ToF16Bits(reflectScale)
					out.BData[base+r*2+0] = byte(bits)
					out.BData[base+r*2+1] = byte(bits >> 8)
					copy(out.BData[base+64+r*32:base+64+(r+1)*32], unsafe.Slice((*byte)(unsafe.Pointer(&aux[0])), 32))
				}
			}
		}
	}
	return out
}
