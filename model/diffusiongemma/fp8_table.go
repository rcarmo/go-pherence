package diffusiongemma

import "math"

var diffusionGemmaFP8E4M3Table = func() [256]float32 {
	var table [256]float32
	for i := 0; i < len(table); i++ {
		table[i] = fp8DecodeE4M3(byte(i))
	}
	return table
}()

// fp8DecodeE4M3 decodes a single FP8 E4M3 byte to float32.
func fp8DecodeE4M3(b byte) float32 {
	sign := float32(1)
	if b&0x80 != 0 {
		sign = -1
	}
	exp := int((b >> 3) & 0x0f)
	mantissa := int(b & 0x07)
	if exp == 0 {
		if mantissa == 0 {
			return sign * 0
		}
		return sign * float32(math.Ldexp(float64(mantissa)/8.0, -6))
	}
	if exp == 0x0f && mantissa == 0x07 {
		return float32(math.NaN())
	}
	return sign * float32(math.Ldexp(1.0+float64(mantissa)/8.0, exp-7))
}
