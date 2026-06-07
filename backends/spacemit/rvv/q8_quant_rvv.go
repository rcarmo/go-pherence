package rvv

var Q8QuantDivisor = float32(127.0)

//go:noescape
func QuantizeQ8Block32RVV(src *float32, dst *byte, divisor *float32)
