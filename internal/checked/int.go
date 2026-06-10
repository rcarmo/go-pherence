package checked

// MaxInt64 returns the largest positive int64 value.
func MaxInt64() int64 { return int64(^uint64(0) >> 1) }

// SaturatingAddInt64 returns 0 for negative inputs, MaxInt64 on overflow, or
// a+b otherwise.
func SaturatingAddInt64(a, b int64) int64 {
	if a < 0 || b < 0 {
		return 0
	}
	max := MaxInt64()
	if a > max-b {
		return max
	}
	return a + b
}

// SaturatingMulInt64 returns 0 for negative inputs, MaxInt64 on overflow, or
// a*b otherwise.
func SaturatingMulInt64(a, b int64) int64 {
	if a < 0 || b < 0 {
		return 0
	}
	max := MaxInt64()
	if b != 0 && a > max/b {
		return max
	}
	return a * b
}

// AddInt returns a+b and false when either input is negative or the sum would
// overflow int.
func AddInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt-b {
		return 0, false
	}
	return a + b, true
}

// MulInt returns a*b and false when either input is negative or the product
// would overflow int.
func MulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}
