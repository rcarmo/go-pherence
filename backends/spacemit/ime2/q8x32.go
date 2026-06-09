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

// PackF32ToQ80x32RowScale packs weights into the native A100 Q8_0 x32 layout
// using one scale for each full output row, repeated across all K32 blocks. This
// mirrors the row-global quantization contract used by the existing Whisper
// native int8 path and is useful for compatibility experiments.
func PackF32ToQ80x32RowScale(M, K int, f32 []float32) Q80x32 {
	if M%32 != 0 || K%32 != 0 || len(f32) < M*K {
		return Q80x32{M: M, K: K}
	}
	groups, subs := M/32, K/32
	out := make([]byte, groups*subs*K3I8I8BTileBytes)
	sc := make([]float32, M)
	for row := 0; row < M; row++ {
		maxAbs := float32(0)
		base := row * K
		for k := 0; k < K; k++ {
			v := float32(math.Abs(float64(f32[base+k])))
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs != 0 {
			sc[row] = maxAbs / 127.0
		}
	}
	for g := 0; g < groups; g++ {
		for sb := 0; sb < subs; sb++ {
			block := out[(g*subs+sb)*K3I8I8BTileBytes:]
			scales := block[:64]
			qs := block[64 : 64+1024]
			for r := 0; r < 32; r++ {
				row := g*32 + r
				base := row*K + sb*32
				d := sc[row]
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

// QuantizeF32RowsQ8M4GELURowScaleInto is the row-global-scale form of
// QuantizeF32RowsQ8M4GELUInto. It repeats each row's full-K GELU scale across
// all K32 A blocks to mimic Whisper's native int8 activation quantization.
func QuantizeF32RowsQ8M4GELURowScaleInto(rows [4][]float32, kBlks int, dst []byte) []byte {
	out := dst[:kBlks*K3I8I8ABlockM4Bytes]
	var sc [4]float32
	for r := 0; r < 4; r++ {
		maxAbs := float32(0)
		for k := 0; k < kBlks*32; k++ {
			v := float32(math.Abs(float64(geluQ8(rows[r][k]))))
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs != 0 {
			sc[r] = maxAbs / 127.0
		}
	}
	for sb := 0; sb < kBlks; sb++ {
		dstOff := sb * K3I8I8ABlockM4Bytes
		for r := 0; r < 4; r++ {
			scale := sc[r]
			inv := float32(0)
			if scale != 0 {
				inv = 1 / scale
			}
			binary.LittleEndian.PutUint32(out[dstOff+r*4:], math.Float32bits(scale))
			sum := 0
			for k := 0; k < 32; k++ {
				q := int(math.Round(float64(geluQ8(rows[r][sb*32+k]) * inv)))
				if q > 127 {
					q = 127
				}
				if q < -128 {
					q = -128
				}
				out[dstOff+24+r*32+k] = byte(int8(q))
				sum += q
			}
			binary.LittleEndian.PutUint16(out[dstOff+16+r*2:], uint16(int16(-sum)))
		}
	}
	return out
}

// QuantizeF32RowsQ8M4 packs four activation rows [4,K] into the M4 A layout
// consumed by K3I8I8M4: per K32 block fp32 scale[4], int16 negative_sum[4],
// int8 q[4][32].
func QuantizeF32RowsQ8M4(rows [4][]float32, kBlks int) []byte {
	return QuantizeF32RowsQ8M4Into(rows, kBlks, make([]byte, kBlks*K3I8I8ABlockM4Bytes))
}

// QuantizeF32RowsQ8M4Into is the allocation-free form of QuantizeF32RowsQ8M4.
// dst must have at least kBlks*K3I8I8ABlockM4Bytes bytes and is returned sliced
// to that length. This is intended for persistent A100 worker scratch buffers.
func QuantizeF32RowsQ8M4Into(rows [4][]float32, kBlks int, dst []byte) []byte {
	out := dst[:kBlks*K3I8I8ABlockM4Bytes]
	for sb := 0; sb < kBlks; sb++ {
		dstOff := sb * K3I8I8ABlockM4Bytes
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
			binary.LittleEndian.PutUint32(out[dstOff+r*4:], math.Float32bits(scale))
			sum := 0
			for k := 0; k < 32; k++ {
				q := int(math.Round(float64(rows[r][sb*32+k] * inv)))
				if q > 127 {
					q = 127
				}
				if q < -128 {
					q = -128
				}
				out[dstOff+24+r*32+k] = byte(int8(q))
				sum += q
			}
			binary.LittleEndian.PutUint16(out[dstOff+16+r*2:], uint16(int16(-sum)))
		}
	}
	return out
}

func fastTanhQ8(x float32) float32 {
	if x > 4.97 {
		return 1
	}
	if x < -4.97 {
		return -1
	}
	x2 := x * x
	a := x * (135135 + x2*(17325+x2*(378+x2)))
	b := float32(135135) + x2*(62370+x2*(3150+x2*28))
	return a / b
}

func geluQ8(v float32) float32 {
	const c = float32(0.7978845608028654)
	inner := c * (v + 0.044715*v*v*v)
	return 0.5 * v * (1 + fastTanhQ8(inner))
}

// QuantizeF32RowsQ8M4GELUInto is like QuantizeF32RowsQ8M4Into, but applies the
// Whisper GELU approximation while quantizing. It is intended for fused
// FFN paths that feed FC1 output directly into an A100 FC2 kernel without a
// separate GELU pass over the full hidden matrix.
func QuantizeF32RowsQ8M4GELUInto(rows [4][]float32, kBlks int, dst []byte) []byte {
	out := dst[:kBlks*K3I8I8ABlockM4Bytes]
	for sb := 0; sb < kBlks; sb++ {
		dstOff := sb * K3I8I8ABlockM4Bytes
		for r := 0; r < 4; r++ {
			var vals [32]float32
			maxAbs := float32(0)
			for k := 0; k < 32; k++ {
				v := geluQ8(rows[r][sb*32+k])
				vals[k] = v
				av := float32(math.Abs(float64(v)))
				if av > maxAbs {
					maxAbs = av
				}
			}
			scale := float32(0)
			inv := float32(0)
			if maxAbs != 0 {
				scale = maxAbs / 127.0
				inv = 1 / scale
			}
			binary.LittleEndian.PutUint32(out[dstOff+r*4:], math.Float32bits(scale))
			sum := 0
			for k := 0; k < 32; k++ {
				q := int(math.Round(float64(vals[k] * inv)))
				if q > 127 {
					q = 127
				}
				if q < -128 {
					q = -128
				}
				out[dstOff+24+r*32+k] = byte(int8(q))
				sum += q
			}
			binary.LittleEndian.PutUint16(out[dstOff+16+r*2:], uint16(int16(-sum)))
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
