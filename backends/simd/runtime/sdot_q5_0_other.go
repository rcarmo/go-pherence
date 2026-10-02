//go:build !amd64

package simd

// HasSdotQ5_0Asm is false: no fused Q5_0 kernel on this architecture.
var HasSdotQ5_0Asm = false

func sdotQ5_0Asm(x []float32, raw []byte, blocks int) float32 { return SdotQ5_0Ref(x, raw) }
