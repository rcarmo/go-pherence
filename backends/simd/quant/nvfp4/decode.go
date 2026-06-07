package nvfp4

import "github.com/rcarmo/go-pherence/backends/simd/quant/fp8"

// DecodeFP4E2M1 decodes NVIDIA FP4 E2M1 values as used by NVFP4 packed
// weights. Codes 0..7 map to positive {0, 0.5, 1, 1.5, 2, 3, 4, 6}; bit 3 is
// the sign bit.
func DecodeFP4E2M1(code byte) float32 {
	mag := [...]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}[code&0x7]
	if code&0x8 != 0 {
		return -mag
	}
	return mag
}

// DecodeF8E4M3 decodes safetensors F8_E4M3FN scale bytes. This finite-only
// E4M3 variant has bias 7, subnormals at exponent field 0, no infinities, and
// reserves only all-ones exponent+mantissa as NaN.
//
// NVFP4 uses E4M3 as its per-block scale format, so this delegates to the
// canonical (LUT-backed) decoder in backends/simd/quant/fp8 — proven
// bit-identical over all 256 codes.
func DecodeF8E4M3(code byte) float32 {
	return fp8.DecodeE4M3(code)
}

// UnpackNVFP4 expands packed low-nibble-first FP4 bytes into decoded E2M1
// values. It is primarily a test/prototype helper; production paths should
// dequantize directly from packed bytes.
func UnpackNVFP4(packed []byte, count int) []float32 {
	if count < 0 || countExceedsPackedNibbles(count, len(packed)) {
		return nil
	}
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		b := packed[i/2]
		code := b & 0x0f
		if i%2 == 1 {
			code = b >> 4
		}
		out[i] = DecodeFP4E2M1(code)
	}
	return out
}
