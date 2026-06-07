package k3

var q8QuantDivisor = float32(127.0)

//go:noescape
func quantizeQ8Block32RVV(src *float32, dst *byte, divisor *float32)
