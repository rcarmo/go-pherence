// Package half provides IEEE-754 half-precision (FP16) and bfloat16 (BF16)
// to float32 conversion.
//
// These conversions were previously duplicated across loader/gguf, model, and
// model/ideogram4. The three independent FP16 implementations were proven
// bit-equivalent on all 65536 inputs for every finite/inf/zero value (only NaN
// bit-payloads differed, which is harmless), so they were consolidated here.
//
// half imports only the standard library, so any package may depend on it
// without risking an import cycle.
package half

import "math"

// F16ToF32 converts an IEEE-754 half-precision value to float32 (no unsafe).
func F16ToF32(u uint16) float32 {
	sign := uint32(u >> 15)
	exp := uint32((u >> 10) & 0x1F)
	mant := uint32(u & 0x3FF)
	if exp == 0x1F {
		// inf or NaN (mantissa preserved)
		return math.Float32frombits(sign<<31 | 0x7F800000 | mant<<13)
	}
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31) // ±0
		}
		// subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	}
	return math.Float32frombits(sign<<31 | (exp+112)<<23 | mant<<13)
}

// BF16ToF32 converts a bfloat16 value (the high 16 bits of a float32) to float32.
func BF16ToF32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}
