package simd

import (
	"encoding/binary"

	"github.com/rcarmo/go-pherence/half"
)

// Q5_0Block is the ggml Q5_0 block size: f16 scale, 32 high bits, 16 nibble bytes.
const Q5_0BlockBytes = 22

// SdotQ5_0 returns Sdot(x, w) where w is the row stored as ggml Q5_0 blocks
// (w[i] = f16(d) * float32(q[i]-16)), without materialising w. On admitted
// hardware the fused kernel dequantises each value with the same single
// rounding and accumulates in Sdot's exact AVX2 order, so results are bit-equal
// to Sdot over the widened row. Callers must check HasSdotQ5_0Asm before relying
// on its speed; elsewhere this computes the reference (widen then Sdot).
// Requires len(raw) == len(x)/32*22 and len(x)%32 == 0.
func SdotQ5_0(x []float32, raw []byte) float32 {
	blocks := len(x) / 32
	if len(x)%32 != 0 || len(raw) != blocks*Q5_0BlockBytes {
		panic("simd: SdotQ5_0 shape")
	}
	if HasSdotQ5_0Asm && blocks > 0 {
		return sdotQ5_0Asm(x, raw, blocks)
	}
	return SdotQ5_0Ref(x, raw)
}

// SdotQ5_0Ref widens the Q5_0 row exactly as the loader does, then calls Sdot.
func SdotQ5_0Ref(x []float32, raw []byte) float32 {
	w := make([]float32, len(x))
	DequantQ5_0Into(w, raw)
	return Sdot(x, w)
}

// DequantQ5_0Into widens Q5_0 blocks with the loader's exact expression.
func DequantQ5_0Into(dst []float32, raw []byte) {
	if len(raw)%Q5_0BlockBytes != 0 || len(dst) != len(raw)/Q5_0BlockBytes*32 {
		panic("simd: DequantQ5_0Into shape")
	}
	for b := 0; b < len(raw)/Q5_0BlockBytes; b++ {
		block := raw[b*Q5_0BlockBytes:][:Q5_0BlockBytes]
		d := half.F16ToF32(binary.LittleEndian.Uint16(block))
		high := binary.LittleEndian.Uint32(block[2:])
		for i := 0; i < 32; i++ {
			q := uint32(block[6+i%16])
			if i < 16 {
				q &= 15
			} else {
				q >>= 4
			}
			q |= ((high >> uint(i)) & 1) << 4
			dst[b*32+i] = d * float32(int(q)-16)
		}
	}
}
