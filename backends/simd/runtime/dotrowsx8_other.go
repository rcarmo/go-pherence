//go:build !amd64

package simd

func dotRowsx8(out, w, x []float32, cols int) {
	d0, d1, d2, d3 := dotRowsx4(w[:4*cols], x, cols)
	out[0], out[1], out[2], out[3] = d0, d1, d2, d3
	d0, d1, d2, d3 = dotRowsx4(w[4*cols:8*cols], x, cols)
	out[4], out[5], out[6], out[7] = d0, d1, d2, d3
}
