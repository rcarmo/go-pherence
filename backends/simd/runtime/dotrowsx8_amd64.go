//go:build amd64

package simd

//go:noescape
func dotRowsx8Asm(out []float32, w []float32, x []float32, cols int)

func dotRowsx8(out, w, x []float32, cols int) {
	// The assembly loop has no tail path. The measured model-convolution
	// reductions are multiples of16 and at most2304; preserve larger generic
	// projection widths and all other shapes through the established x4 path.
	if HasDotAsm && cols%16 == 0 && cols <= 2304 {
		dotRowsx8Asm(out[:8], w[:8*cols], x[:cols], cols)
		return
	}
	d0, d1, d2, d3 := dotRowsx4(w[:4*cols], x, cols)
	out[0], out[1], out[2], out[3] = d0, d1, d2, d3
	d0, d1, d2, d3 = dotRowsx4(w[4*cols:8*cols], x, cols)
	out[4], out[5], out[6], out[7] = d0, d1, d2, d3
}
