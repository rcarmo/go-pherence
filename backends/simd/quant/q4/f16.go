package q4

import "math"

// Float16ToFloat32 converts a uint16 IEEE 754 half-precision to float32.
func Float16ToFloat32(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := int((h >> 10) & 0x1F)
	frac := int(h & 0x03FF)
	if exp == 0 {
		if frac == 0 {
			if sign < 0 {
				return math.Float32frombits(0x80000000)
			}
			return 0
		}
		return sign * float32(math.Ldexp(float64(frac), -24))
	}
	if exp == 0x1F {
		if frac == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	return sign * float32(math.Ldexp(1+float64(frac)/1024, exp-15))
}
