package ime2

import (
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
)

const K3I8I8BTileBytes = 64 + 1024 // fp16 scale[32] + int8 q[32][32]
const K3I8I8ABlockM1Bytes = 38
const K3I8I8ABlockM4Bytes = 16 + 8 + 128

// Q80x32 stores f32 weights quantized as Q8_0 in the native K3 A100 N32/K32
// layout consumed by K3I8I8M1/M4. M is the output dimension (rows), K input.
type Q80x32 struct {
	M, K  int
	BData []byte
	Valid bool
}

// PackF32ToQ80x32 packs weights W[M,K] into the native A100 Q8_0 x32 layout.
// Requires M%32==0 and K%32==0.
func PackF32ToQ80x32(M, K int, f32 []float32) Q80x32 {
	if M%32 != 0 || K%32 != 0 || len(f32) < M*K {
		return Q80x32{M: M, K: K}
	}
	groups, subs := M/32, K/32
	out := make([]byte, groups*subs*K3I8I8BTileBytes)
	for g := 0; g < groups; g++ {
		for sb := 0; sb < subs; sb++ {
			block := out[(g*subs+sb)*K3I8I8BTileBytes:]
			scales := block[:64]
			qs := block[64 : 64+1024]
			for r := 0; r < 32; r++ {
				row := g*32 + r
				base := row*K + sb*32
				maxAbs := float32(0)
				for k := 0; k < 32; k++ {
					v := float32(math.Abs(float64(f32[base+k])))
					if v > maxAbs {
						maxAbs = v
					}
				}
				d := float32(0)
				if maxAbs != 0 {
					d = maxAbs / 127.0
				}
				binary.LittleEndian.PutUint16(scales[r*2:], half.F32ToF16(d))
				inv := float32(0)
				if d != 0 {
					inv = 1 / d
				}
				for k := 0; k < 32; k++ {
					q := int(math.Round(float64(f32[base+k] * inv)))
					if q > 127 {
						q = 127
					}
					if q < -128 {
						q = -128
					}
					qs[r*32+k] = byte(int8(q))
				}
			}
		}
	}
	return Q80x32{M: M, K: K, BData: out, Valid: true}
}

// QuantizeF32RowsQ8M4 packs four activation rows [4,K] into the M4 A layout
// consumed by K3I8I8M4: per K32 block fp32 scale[4], int16 negative_sum[4],
// int8 q[4][32].
func QuantizeF32RowsQ8M4(rows [4][]float32, kBlks int) []byte {
	out := make([]byte, kBlks*K3I8I8ABlockM4Bytes)
	for sb := 0; sb < kBlks; sb++ {
		dst := sb * K3I8I8ABlockM4Bytes
		for r := 0; r < 4; r++ {
			maxAbs := float32(0)
			for k := 0; k < 32; k++ {
				v := float32(math.Abs(float64(rows[r][sb*32+k])))
				if v > maxAbs {
					maxAbs = v
				}
			}
			scale := float32(0)
			inv := float32(0)
			if maxAbs != 0 {
				scale = maxAbs / 127.0
				inv = 1 / scale
			}
			binary.LittleEndian.PutUint32(out[dst+r*4:], math.Float32bits(scale))
			sum := 0
			for k := 0; k < 32; k++ {
				q := int(math.Round(float64(rows[r][sb*32+k] * inv)))
				if q > 127 {
					q = 127
				}
				if q < -128 {
					q = -128
				}
				out[dst+24+r*32+k] = byte(int8(q))
				sum += q
			}
			binary.LittleEndian.PutUint16(out[dst+16+r*2:], uint16(int16(-sum)))
		}
	}
	return out
}

// K3I8I8 dispatches the native A100 Q8_0 x Q8_0 kernel across N32 tiles. c is
// row-major [countM, ldc]. Returns the number of M rows consumed (4 or 1).
func K3I8I8(a, b *byte, c *float32, countM, countN, kBlks, ldc int) int {
	if countM >= 4 {
		for n := 0; n < countN; n += 32 {
			K3I8I8M4(a, (*byte)(unsafe.Add(unsafe.Pointer(b), n/32*kBlks*K3I8I8BTileBytes)), (*float32)(unsafe.Add(unsafe.Pointer(c), n*4)), kBlks, ldc*4)
		}
		return 4
	}
	for n := 0; n < countN; n += 32 {
		K3I8I8M1(a, (*byte)(unsafe.Add(unsafe.Pointer(b), n/32*kBlks*K3I8I8BTileBytes)), (*float32)(unsafe.Add(unsafe.Pointer(c), n*4)), kBlks, 32)
	}
	return 1
}
