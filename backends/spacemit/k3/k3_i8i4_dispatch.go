package k3

import "unsafe"

const k3I8I4BTileBytes = 608 // native PR#22863: fp16 d[32]=64B + qs[512]=512B, no ZP section
const k3I8I4ABlockM1Bytes = 38
const k3I8I4ABlockM4Bytes = 16 + 8 + 128

// k3I8I4 dispatches like llama.cpp's gemm_kernel_i8i4(): use the M4 kernel
// when at least four A rows are available, otherwise use M1. countN must be a
// multiple of 32 and B must be laid out as contiguous N32 tiles, each with
// kBlks 608-byte Q4_1x32 subblocks. ldc is in float elements.
func k3I8I4(a, b *byte, c *float32, countM, countN, kBlks, ldc int) int {
	if countM >= 4 {
		for n := 0; n < countN; n += 32 {
			k3I8I4M4(
				a,
				(*byte)(unsafe.Add(unsafe.Pointer(b), n/32*kBlks*k3I8I4BTileBytes)),
				(*float32)(unsafe.Add(unsafe.Pointer(c), n*4)),
				kBlks,
				ldc*4,
			)
		}
		return 4
	}
	for n := 0; n < countN; n += 32 {
		k3I8I4M1(
			a,
			(*byte)(unsafe.Add(unsafe.Pointer(b), n/32*kBlks*k3I8I4BTileBytes)),
			(*float32)(unsafe.Add(unsafe.Pointer(c), n*4)),
			kBlks,
			32,
		)
	}
	return 1
}

// packQ8RowsM4 packs four M1 Q8 rows into the A layout consumed by k3I8I4M4.
// Each input row is kBlks*[fp32 scale][int16 sum][32 int8 q].
func packQ8RowsM4(rows [4][]byte, kBlks int) []byte {
	out := make([]byte, kBlks*k3I8I4ABlockM4Bytes)
	for sb := 0; sb < kBlks; sb++ {
		dst := sb * k3I8I4ABlockM4Bytes
		for r := 0; r < 4; r++ {
			src := sb * k3I8I4ABlockM1Bytes
			copy(out[dst+r*4:dst+r*4+4], rows[r][src:src+4])
			copy(out[dst+16+r*2:dst+16+r*2+2], rows[r][src+4:src+6])
			copy(out[dst+24+r*32:dst+24+(r+1)*32], rows[r][src+6:src+38])
		}
	}
	return out
}
