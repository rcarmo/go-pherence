package nvfp4

import "math"

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
func DecodeF8E4M3(code byte) float32 {
	sign := code & 0x80
	exp := (code >> 3) & 0x0f
	mant := code & 0x07
	var v float32
	if exp == 0 {
		if mant == 0 {
			v = 0
		} else {
			v = float32(mant) / 8 * float32(math.Ldexp(1, -6))
		}
	} else if exp == 0x0f && mant == 0x07 {
		v = float32(math.NaN())
	} else {
		v = (1 + float32(mant)/8) * float32(math.Ldexp(1, int(exp)-7))
	}
	if sign != 0 {
		return -v
	}
	return v
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
