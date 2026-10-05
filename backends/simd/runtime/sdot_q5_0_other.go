//go:build !amd64

package simd

// HasSdotQ5_0Asm is false: no fused Q5_0 kernel on this architecture.
var HasSdotQ5_0Asm = false

func sdotQ5_0Asm(x []float32, raw []byte, blocks int) float32 { return SdotQ5_0Ref(x, raw) }

func sdotQ5_0x4Asm(x []float32, stride int, raw []byte, blocks int, out *[4]float32) {
	n := blocks * 32
	for r := range out {
		out[r] = SdotQ5_0Ref(x[r*stride:r*stride+n], raw)
	}
}
