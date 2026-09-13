package omnivoice

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

// snakeChannel updates a channel with bounded scratch. Separate vector operations
// preserve the scalar expression's float32 rounding boundaries; only sine uses
// a bounded approximation on supported CPUs. Scratch stays on the caller stack.
func snakeChannel(row []float32, alpha float32) {
	var storage [256]float32
	scale := float32(1) / (alpha + 1e-9)
	for start := 0; start < len(row); start += len(storage) {
		x := row[start:min(start+len(storage), len(row))]
		tmp := storage[:len(x)]
		simd.VecScale(tmp, x, alpha)
		simd.SinF32To(tmp, tmp)
		simd.VecMul(tmp, tmp, tmp)
		simd.VecScale(tmp, tmp, scale)
		simd.VecAdd(x, x, tmp)
	}
}
